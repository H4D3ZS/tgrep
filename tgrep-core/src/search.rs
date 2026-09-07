//! A minimal, in-process "directory + pattern -> matches" library entry
//! point — the piece `tgrep-core` was missing.
//!
//! `matching.rs` (moved here from the CLI crate) already does the real
//! per-file work — compile a pattern once, scan a file's text, produce
//! line-level hits — entirely independent of how the result gets used.
//! `tgrep-cli`'s `search.rs`/`serve.rs` wrap that in ~3,700 lines of output
//! formatting (`--json`, `--vimgrep`, columns, context separators, sort
//! modes, the client/server RPC…) that a library caller doesn't want.
//!
//! This module is the other wrapper: walk a directory (`walker`, already
//! ignore-aware and binary-detecting), read each candidate file, and run it
//! through the shared matcher — returning a plain `Vec<Match>`, no writer, no
//! process, no JSON round-trip. It intentionally covers the common case
//! (case-insensitive/fixed-string literal-or-regex search over a directory or
//! a single file) rather than the CLI's full flag surface; a caller that
//! needs sort modes, glob/type filters, or streaming output should still go
//! through `tgrep-cli`.

use std::path::Path;

use anyhow::Result;

use crate::matching::{build_search_matcher, FileMatches, MatchOptions, MatcherConfig};
use crate::walker::{walk_dir, WalkOptions};

/// One line-level match: which file, which line, and the line's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// Path relative to the search root (forward-slash separated, even on
    /// Windows, matching ripgrep/tgrep-cli's own display convention).
    pub path: String,
    /// 1-based line number.
    pub line: usize,
    /// The matching line's text, trimmed of its line terminator.
    pub content: String,
}

/// Options for [`search_directory`]. Deliberately small — see the module docs
/// for what's out of scope.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub case_insensitive: bool,
    /// Treat `pattern` as a literal string rather than a regex.
    pub fixed_string: bool,
    /// Stop once this many matches have been collected. `0` means unbounded.
    pub max_results: usize,
    /// Files above this size are skipped. `None` disables the limit.
    pub max_file_size: Option<u64>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            case_insensitive: false,
            fixed_string: false,
            max_results: 0,
            max_file_size: Some(64 * 1024 * 1024), // tgrep-cli's own default
        }
    }
}

/// Search `root` (a directory, walked ignore-aware and binary-skipping, or a
/// single file) for `pattern`, in-process — no subprocess, no server. Returns
/// every match found, capped at `opts.max_results` when nonzero.
///
/// Files that fail to decode as UTF-8 (lossily) are skipped, matching the
/// CLI's own behavior for non-text files it can't confidently read.
pub fn search_directory(pattern: &str, root: &Path, opts: &SearchOptions) -> Result<Vec<Match>> {
    let matcher_cfg = MatcherConfig {
        case_insensitive: opts.case_insensitive,
        fixed_string: opts.fixed_string,
        ..Default::default()
    };
    let matcher = build_search_matcher(&[pattern.to_string()], &matcher_cfg)?;
    let match_opts = MatchOptions {
        invert_match: false,
        multiline: false,
        only_matching: false,
        before_context: 0,
        after_context: 0,
        max_count: None,
        passthru: false,
        replace: None,
        stop_on_nonmatch: false,
        vimgrep: false,
        all_spans: false,
    };

    let mut out = Vec::new();
    let cap = if opts.max_results == 0 { usize::MAX } else { opts.max_results };

    let mut scan_one = |path: &Path, rel_display: &str| -> Result<bool> {
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(false); // unreadable — skip, don't fail the whole search
        };
        let text = String::from_utf8_lossy(&bytes);
        let found = FileMatches::find(&text, &matcher, &match_opts)?;
        if found.is_empty() {
            return Ok(false);
        }
        let mut hit_cap = false;
        found.for_each(&match_opts, &matcher, |emit| -> Result<()> {
            if let crate::matching::Emit::Match { line_number, content, .. } = emit {
                out.push(Match {
                    path: rel_display.to_string(),
                    line: line_number,
                    content: content.trim_end_matches(['\n', '\r']).to_string(),
                });
                if out.len() >= cap {
                    hit_cap = true;
                }
            }
            Ok(())
        })?;
        Ok(hit_cap)
    };

    if root.is_file() {
        let display = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        scan_one(root, &display)?;
        return Ok(out);
    }

    let walk_opts = WalkOptions { max_file_size: opts.max_file_size, ..Default::default() };
    let walk = walk_dir(root, &walk_opts);
    for path in &walk.files {
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if scan_one(path, &rel)? {
            break; // cap reached
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn finds_a_real_match_with_line_number() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a.rs", "fn main() {}\nstruct Foo;\n");
        let hits = search_directory("struct Foo", tmp.path(), &SearchOptions::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "a.rs");
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].content, "struct Foo;");
    }

    #[test]
    fn no_match_is_an_empty_vec_not_an_error() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a.rs", "fn main() {}\n");
        let hits = search_directory("NoSuchPattern12345", tmp.path(), &SearchOptions::default()).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn case_insensitive_flag_matters() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a.rs", "let X = 1;\n");
        let sensitive = search_directory("x =", tmp.path(), &SearchOptions::default()).unwrap();
        assert!(sensitive.is_empty());
        let opts = SearchOptions { case_insensitive: true, ..Default::default() };
        let insensitive = search_directory("x =", tmp.path(), &opts).unwrap();
        assert_eq!(insensitive.len(), 1);
    }

    #[test]
    fn fixed_string_treats_regex_metachars_literally() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a.rs", "let v: Vec<String> = vec![];\n");
        let opts = SearchOptions { fixed_string: true, ..Default::default() };
        let hits = search_directory("Vec<String>", tmp.path(), &opts).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn respects_gitignore_via_the_shared_walker() {
        // The walker matches ripgrep's own default: .gitignore only applies
        // inside an actual git repo (`require_git`), so a bare `.gitignore`
        // with no `.git` directory is correctly treated as plain text, not
        // rules. A real `.git` dir (even an empty one) is what makes this
        // test genuine.
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join(".git")).unwrap();
        write(tmp.path(), ".gitignore", "ignored/\n");
        write(tmp.path(), "ignored/secret.rs", "struct Secret;\n");
        write(tmp.path(), "kept.rs", "struct Kept;\n");
        let hits = search_directory("struct ", tmp.path(), &SearchOptions::default()).unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"kept.rs"));
        assert!(!paths.iter().any(|p| p.contains("secret")));
    }

    #[test]
    fn searches_a_single_file_directly() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "only.rs", "const X: i32 = 42;\n");
        let file = tmp.path().join("only.rs");
        let hits = search_directory("const X", &file, &SearchOptions::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "only.rs");
    }

    #[test]
    fn max_results_caps_the_output() {
        let tmp = TempDir::new().unwrap();
        for i in 0..20 {
            write(tmp.path(), &format!("f{i}.rs"), "target line\n");
        }
        let opts = SearchOptions { max_results: 5, ..Default::default() };
        let hits = search_directory("target", tmp.path(), &opts).unwrap();
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn multiple_matches_in_one_file_all_reported() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a.rs", "TODO one\nfine\nTODO two\n");
        let hits = search_directory("TODO", tmp.path(), &SearchOptions::default()).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[1].line, 3);
    }
}
