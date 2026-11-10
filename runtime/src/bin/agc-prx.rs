//! Writes and checks the Agc export shells (PLAN P3.4, decisions D7).
//!
//! usage:
//!   agc-prx write DIR   write libSceAgc.prx and the other four shells into DIR
//!   agc-prx check DIR   load every shell in DIR as AnyPS5's entry stub does and check that each
//!                       export reaches DIR\agcdrv.dll; also call the runtime's probe export
//!                       through a shell written for that purpose

use agcdrv as runtime;
use runtime::agc::exports::{self, MODULES};
use runtime::agc::nid::nid;
use runtime::agc::shell::{self, Dependency, Forward};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn dependency() -> Dependency<'static> {
    Dependency { dll: exports::RUNTIME_DLL, symbol: exports::RUNTIME_ANCHOR }
}

fn write_shell<'a>(path: &Path, functions: impl IntoIterator<Item = &'a str>) -> Result<usize, String> {
    let names: Vec<(String, String)> = functions
        .into_iter()
        .map(|function| (nid(function), format!("{}.{function}", exports::RUNTIME_MODULE)))
        .collect();
    let forwards: Vec<Forward> = names
        .iter()
        .map(|(name, target)| Forward { name, target })
        .collect();
    let file = path.file_name().and_then(|name| name.to_str()).unwrap_or("shell.prx");
    let image = shell::build(file, &forwards, Some(dependency())).map_err(|error| error.to_string())?;
    std::fs::write(path, image).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(forwards.len())
}

fn write(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for module in MODULES {
        let count = write_shell(&dir.join(module.file), module.functions())?;
        println!("{}: {count} exports", module.file);
    }
    Ok(())
}

#[cfg(windows)]
fn check(dir: &Path) -> Result<(), String> {
    use shell::loader::{loaded_symbol, Module};
    let absolute = std::fs::canonicalize(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    // The canonical form carries the \\?\ prefix; the entry stub passes plain absolute paths.
    let text = absolute.to_string_lossy();
    let absolute = PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text));
    if !absolute.join(exports::RUNTIME_DLL).is_file() {
        return Err(format!("{} has no {}", absolute.display(), exports::RUNTIME_DLL));
    }

    // The probe: a shell forwarding one NID to the runtime, called through the resolved address.
    let probe = absolute.join("agc-shell-probe.prx");
    write_shell(&probe, [exports::PROBE])?;
    let module = Module::load(&probe)?;
    let address = module
        .symbol(&nid(exports::PROBE))
        .ok_or("the probe shell does not resolve its export")?;
    // SAFETY: the forwarder resolves to `mesa_agc_shell_probe`, a sysv64 function without arguments.
    let value = unsafe { std::mem::transmute::<usize, extern "sysv64" fn() -> u32>(address)() };
    if value != exports::PROBE_VALUE {
        return Err(format!("probe returned 0x{value:x}, expected 0x{:x}", exports::PROBE_VALUE));
    }
    let direct = loaded_symbol(exports::RUNTIME_DLL, exports::PROBE);
    if direct != Some(address) {
        return Err(format!("probe resolved to 0x{address:x}, the runtime exports it at {direct:x?}"));
    }
    drop(module);
    let _ = std::fs::remove_file(&probe);
    println!("probe: {} reached through a shell (0x{address:x})", exports::PROBE);

    let mut failures = 0;
    for module in MODULES {
        let path = absolute.join(module.file);
        let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let exported = shell::read_exports(&image).map_err(|error| format!("{}: {error}", path.display()))?;
        let loaded = Module::load(&path)?;
        let mut reached = 0;
        for (name, target) in &exported {
            let symbol = target.split_once('.').map(|(_, symbol)| symbol).unwrap_or(target);
            let expected = loaded_symbol(exports::RUNTIME_DLL, symbol);
            match (loaded.symbol(name), expected) {
                (Some(actual), Some(expected)) if actual == expected => reached += 1,
                (actual, expected) => {
                    failures += 1;
                    eprintln!("{}: {name} -> {target}: resolved {actual:x?}, runtime {expected:x?}", module.file);
                }
            }
        }
        println!("{}: {reached} of {} exports reach the runtime", module.file, exported.len());
    }
    if failures > 0 {
        return Err(format!("{failures} exports do not reach the runtime"));
    }
    Ok(())
}

#[cfg(not(windows))]
fn check(_dir: &Path) -> Result<(), String> {
    Err("check needs the Windows loader".into())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let result = match (args.get(1).map(String::as_str), args.get(2)) {
        (Some("write"), Some(dir)) => write(Path::new(dir)),
        (Some("check"), Some(dir)) => check(Path::new(dir)),
        _ => Err("usage: agc-prx write DIR | agc-prx check DIR".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("agc-prx: {error}");
            ExitCode::FAILURE
        }
    }
}
