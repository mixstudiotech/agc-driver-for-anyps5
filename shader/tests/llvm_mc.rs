//! llvm-mc oracle for the translator's encodings.
//!
//! `llvm_mc/cases.txt` holds guest programs assembled by llvm-mc (written by
//! `driver/tools/isa-tables/llvm_mc_cases.py`). Every translation must equal the words recorded
//! in `llvm_mc/golden/<cpu>.txt`, beside llvm-mc's disassembly of them, so the bytes are
//! checked without LLVM. With `ANYPS5_LLVM_MC` naming an llvm-mc binary the test also
//! disassembles every translation and requires zero invalid words. The recorded text is for
//! inspection and may change with LLVM's formatting across releases. Adding
//! `ANYPS5_UPDATE_GOLDEN=1` rewrites the golden files instead of comparing.

#[path = "llvm_mc/corpus.rs"]
mod corpus;

use corpus::{cases, golden_path, read_golden, translate, Case, Golden, TARGETS};
use shader::TargetIsa;
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

fn write_golden(cpu: &str, entries: &BTreeMap<String, Golden>, order: &[Case]) {
    let mut out = format!(
        "# Translations for {cpu} with llvm-mc's disassembly; see shader/tests/llvm_mc.rs.\n"
    );
    for case in order {
        let entry = &entries[&case.name];
        out.push_str(&format!("## {}\n", case.name));
        match &entry.words {
            Ok(words) => {
                let hex: Vec<String> = words.iter().map(|w| format!("{w:08x}")).collect();
                out.push_str(&format!("words: {}\n", hex.join(" ")));
            }
            Err(error) => out.push_str(&format!("error: {error}\n")),
        }
        for line in &entry.text {
            out.push_str(&format!("    {line}\n"));
        }
    }
    std::fs::create_dir_all(golden_path(cpu).parent().unwrap()).unwrap();
    std::fs::write(golden_path(cpu), out).unwrap();
}

/// llvm-mc's disassembly of `words` and the number of words it found invalid.
fn disassemble(mc: &std::ffi::OsStr, cpu: &str, words: &[u32]) -> (Vec<String>, usize) {
    let bytes: Vec<String> = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| format!("{b:#04x}"))
        .collect();
    let mut child = Command::new(mc)
        .args(["-triple=amdgcn", &format!("-mcpu={cpu}"), "--disassemble"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run llvm-mc");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(bytes.join(" ").as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "llvm-mc failed for {cpu}: {stderr}");
    let invalid = stderr.matches("invalid instruction encoding").count();
    let text = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('.'))
        .map(str::to_string)
        .collect();
    (text, invalid)
}

/// The differential executor runs translations through `HostDecoder`: every program it can
/// meet decodes back to gfx1013 instructions.
#[test]
fn translations_decode_back_for_the_reference_executor() {
    let mut failures = Vec::new();
    for case in cases() {
        for target in [TargetIsa::Gfx103, TargetIsa::Gfx11, TargetIsa::Gfx115] {
            let Ok(words) = translate(&case, target) else {
                continue;
            };
            let decoded = shader::decode::host::HostDecoder::new(target)
                .and_then(|d| d.decode(&words, case.stage));
            if let Err(e) = decoded {
                failures.push(format!("{target:?} {}: {e}", case.name));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn translations_match_the_llvm_mc_golden_files() {
    let mc = std::env::var_os("ANYPS5_LLVM_MC");
    let update = std::env::var_os("ANYPS5_UPDATE_GOLDEN").is_some();
    assert!(
        !update || mc.is_some(),
        "ANYPS5_UPDATE_GOLDEN needs ANYPS5_LLVM_MC"
    );
    let cases = cases();
    let mut failures = Vec::new();
    for (target, cpu) in TARGETS {
        let golden = read_golden(cpu);
        let mut fresh = BTreeMap::new();
        for case in &cases {
            let recorded = golden.get(&case.name).cloned().unwrap_or_default();
            let words = translate(case, target);
            let text = match (&mc, &words) {
                (Some(mc), Ok(words)) => {
                    let (text, invalid) = disassemble(mc, cpu, words);
                    if invalid > 0 {
                        failures.push(format!("{cpu} {}: {invalid} invalid words", case.name));
                    }
                    text
                }
                (Some(_), Err(_)) => Vec::new(),
                (None, _) => recorded.text.clone(),
            };
            let entry = Golden { words, text };
            if !update && entry.words != recorded.words {
                failures.push(format!(
                    "{cpu} {}: got {:08x?}, golden {:08x?}",
                    case.name, entry.words, recorded.words
                ));
            }
            fresh.insert(case.name.clone(), entry);
        }
        if update {
            write_golden(cpu, &fresh, &cases);
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
