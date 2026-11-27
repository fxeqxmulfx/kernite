//! What an editor shows where a kernel names something. rust-analyzer runs over a fixture as the user's editor runs it:
//! the stable toolchain's, with `RUSTUP_TOOLCHAIN=stable` in its environment, which builds the macros with the
//! toolchain that built the tests and runs them in its own proc-macro server; or the one `$RUST_ANALYZER` gives. It is
//! asked, by the language server protocol, what a hover shows at a name in the fixture. A test skips, and says so,
//! where there is none.

mod common;

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use common::{BUILD_ENV, ROOT, Result, written};

/// How long rust-analyzer has to say what a test waits for: it builds the macros first, in a minute or two, and only a
/// hang takes this long, so that no test holds how long rust-analyzer takes.
const LIMIT: Duration = Duration::from_secs(900);

/// The rust-analyzer a test runs: `$RUST_ANALYZER`, or the one on the path, if it runs; `None` where there is none.
fn rust_analyzer() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("RUST_ANALYZER") {
        return Some(path.into());
    }
    let mut run = Command::new("rust-analyzer");
    let here = run.arg("--version").env("RUSTUP_TOOLCHAIN", "stable").output().is_ok_and(|o| o.status.success());
    here.then(|| "rust-analyzer".into())
}

/// A rust-analyzer that has loaded a crate, which a test asks for what an editor shows.
struct Editor {
    child: Child,
    input: ChildStdin,
    messages: Receiver<String>,
    /// The `id` of the next request.
    id: u64,
    /// Where it logs what it says to itself.
    log: PathBuf,
    /// The crate's directory, as a URI.
    root: String,
}

impl Editor {
    /// `binary` run over the crate at `dir`, once it has loaded the crate and built its macros.
    fn open(binary: &Path, dir: &Path) -> Result<Editor> {
        let log = dir.join("../rust-analyzer.log");
        let mut run = Command::new(binary);
        run.env("RUSTUP_TOOLCHAIN", "stable").current_dir(dir);
        run.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(File::create(&log)?);
        for var in BUILD_ENV {
            run.env_remove(var);
        }
        let mut child = run.spawn()?;
        let input = child.stdin.take().ok_or("rust-analyzer has no input")?;
        let output = child.stdout.take().ok_or("rust-analyzer has no output")?;
        let root = format!("file://{}", fs::canonicalize(dir)?.display());
        // What an editor says it is and what it wants: the status of the server, and hovers in Markdown. The server
        // builds into a target directory of its own, which each crate shares.
        let target = Path::new(ROOT).join("target-analyzer").display().to_string();
        let options = format!(
            r#"{{"cargo":{{"targetDir":{}}},"cachePriming":{{"enable":false}},"checkOnSave":false}}"#,
            quote(&target)
        );
        let params = format!(
            concat!(
                r#"{{"processId":null,"rootUri":{root},"workspaceFolders":[{{"uri":{root},"name":"fixture"}}],"#,
                r#""capabilities":{{"experimental":{{"serverStatusNotification":true}},"#,
                r#""textDocument":{{"hover":{{"contentFormat":["markdown"]}}}}}},"initializationOptions":{options}}}"#
            ),
            root = format!("{root:?}"),
            options = options,
        );
        let mut editor = Editor { child, input, messages: messages(output), id: 0, log, root };
        editor.request("initialize", &params)?;
        editor.notify("initialized", "{}")?;
        editor.wait("the crate to load", |m| {
            m.get("method").and_then(Json::text) == Some("experimental/serverStatus")
                && m.get("params").and_then(|p| p.get("quiescent")) == Some(&Json::Bool(true))
        })?;
        Ok(editor)
    }

    /// Sends `method` with `params`, a JSON object; returns the id the answer has.
    fn request(&mut self, method: &str, params: &str) -> Result<u64> {
        self.id += 1;
        let id = self.id;
        self.send(&format!(r#"{{"jsonrpc":"2.0","id":{id},"method":{},"params":{params}}}"#, quote(method)))?;
        Ok(id)
    }

    /// Sends `method` with `params`, a JSON object, which has no answer.
    fn notify(&mut self, method: &str, params: &str) -> Result {
        self.send(&format!(r#"{{"jsonrpc":"2.0","method":{},"params":{params}}}"#, quote(method)))
    }

    fn send(&mut self, body: &str) -> Result {
        write!(self.input, "Content-Length: {}\r\n\r\n{body}", body.len())?;
        Ok(self.input.flush()?)
    }

    /// The first message from now that `want` takes. The server's own requests are answered with nothing, as an editor
    /// that has no answer does.
    fn wait(&mut self, what: &str, want: impl Fn(&Json) -> bool) -> Result<Json> {
        let end = Instant::now() + LIMIT;
        loop {
            let left = end.saturating_duration_since(Instant::now());
            let text =
                self.messages.recv_timeout(left).map_err(|e| format!("waiting for {what}: {e}\n{}", self.said()))?;
            let message = Json::parse(&text).ok_or_else(|| format!("a message that is no JSON: {text}"))?;
            if let (Some(id), Some(_)) = (message.get("id"), message.get("method")) {
                self.send(&format!(r#"{{"jsonrpc":"2.0","id":{},"result":null}}"#, id.show()))?;
            } else if let Some(status) = message.get("params").filter(|_| is_status(&message)) {
                let health = status.get("health").and_then(Json::text);
                if status.get("quiescent") == Some(&Json::Bool(true)) && health != Some("ok") {
                    let note = status.get("message").and_then(Json::text).unwrap_or_default();
                    return Err(format!("rust-analyzer is {health:?} of the crate: {note}\n{}", self.said()).into());
                }
                if want(&message) {
                    return Ok(message);
                }
            } else if want(&message) {
                return Ok(message);
            }
        }
    }

    /// What rust-analyzer said to its log.
    fn said(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default().lines().rev().take(30).collect::<Vec<_>>().join("\n")
    }

    /// What a hover shows in `src/lib.rs` at the first `needle` of it, `skip` characters in: the Markdown; or why not.
    fn shown(&mut self, needle: &str, skip: usize) -> Result<String> {
        self.hover("src/lib.rs", needle, skip)?.ok_or_else(|| format!("a hover at `{needle}` shows nothing").into())
    }

    /// What a hover shows in `file` of the crate at the first `needle` of it, `skip` characters in: the Markdown, or
    /// none.
    fn hover(&mut self, file: &str, needle: &str, skip: usize) -> Result<Option<String>> {
        let path = Path::new(self.root.trim_start_matches("file://")).join(file);
        let text = fs::read_to_string(&path)?;
        let (line, column) = place(&text, needle).ok_or_else(|| format!("{file} has no `{needle}`"))?;
        let uri = format!("{}/{file}", self.root);
        let open = format!(
            r#"{{"textDocument":{{"uri":{},"languageId":"rust","version":1,"text":{}}}}}"#,
            quote(&uri),
            quote(&text)
        );
        self.notify("textDocument/didOpen", &open)?;
        let at = format!(
            r#"{{"textDocument":{{"uri":{}}},"position":{{"line":{line},"character":{}}}}}"#,
            quote(&uri),
            column + skip
        );
        let id = self.request("textDocument/hover", &at)?;
        let answer =
            self.wait("a hover", |m| m.get("id") == Some(&Json::Number(id.to_string())) && m.get("method").is_none())?;
        Ok(answer
            .get("result")
            .and_then(|r| r.get("contents"))
            .and_then(|c| c.get("value"))
            .and_then(Json::text)
            .map(String::from))
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        // Its proc-macro server ends with its input.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Whether `message` is the server telling its status.
fn is_status(message: &Json) -> bool {
    message.get("method").and_then(Json::text) == Some("experimental/serverStatus")
}

/// The messages `output` has, each as the text of its body.
fn messages(output: impl Read + Send + 'static) -> Receiver<String> {
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let mut output = BufReader::new(output);
        loop {
            let mut length = None;
            loop {
                let mut line = String::new();
                if output.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let line = line.trim_end();
                if line.is_empty() {
                    break;
                }
                length = line.strip_prefix("Content-Length: ").and_then(|n| n.parse::<usize>().ok()).or(length);
            }
            let mut body = vec![0; length.unwrap_or(0)];
            if output.read_exact(&mut body).is_err() || send.send(String::from_utf8_lossy(&body).into_owned()).is_err()
            {
                return;
            }
        }
    });
    receive
}

/// The line and the column, from 0, of the first `needle` in `text`, which is ASCII.
fn place(text: &str, needle: &str) -> Option<(usize, usize)> {
    assert!(text.is_ascii(), "a position counts characters, which a fixture of ASCII has as bytes");
    text.lines().enumerate().find_map(|(line, text)| text.find(needle).map(|column| (line, column)))
}

/// `text` as a JSON string.
fn quote(text: &str) -> String {
    let mut out = String::from('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON value, as much of one as the protocol's messages need.
#[derive(Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    /// A number, as it is written.
    Number(String),
    Text(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

type Chars<'a> = std::iter::Peekable<std::str::Chars<'a>>;

impl Json {
    /// `text` as a value, if it is one.
    fn parse(text: &str) -> Option<Json> {
        let mut chars = text.chars().peekable();
        let value = Json::value(&mut chars)?;
        Json::blank(&mut chars);
        chars.peek().is_none().then_some(value)
    }

    fn blank(chars: &mut Chars) {
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
    }

    fn value(chars: &mut Chars) -> Option<Json> {
        Json::blank(chars);
        match chars.peek()? {
            '{' => {
                chars.next();
                let mut fields = vec![];
                loop {
                    Json::blank(chars);
                    if chars.next_if_eq(&'}').is_some() {
                        return Some(Json::Object(fields));
                    }
                    if !fields.is_empty() {
                        chars.next_if_eq(&',')?;
                        Json::blank(chars);
                    }
                    let key = Json::string(chars)?;
                    Json::blank(chars);
                    chars.next_if_eq(&':')?;
                    fields.push((key, Json::value(chars)?));
                }
            }
            '[' => {
                chars.next();
                let mut items = vec![];
                loop {
                    Json::blank(chars);
                    if chars.next_if_eq(&']').is_some() {
                        return Some(Json::Array(items));
                    }
                    if !items.is_empty() {
                        chars.next_if_eq(&',')?;
                    }
                    items.push(Json::value(chars)?);
                }
            }
            '"' => Json::string(chars).map(Json::Text),
            _ => {
                let mut word = String::new();
                while let Some(c) = chars.next_if(|c| !",}] \n\t\r".contains(*c)) {
                    word.push(c);
                }
                match word.as_str() {
                    "null" => Some(Json::Null),
                    "true" => Some(Json::Bool(true)),
                    "false" => Some(Json::Bool(false)),
                    _ => word.parse::<f64>().is_ok().then_some(Json::Number(word)),
                }
            }
        }
    }

    fn string(chars: &mut Chars) -> Option<String> {
        chars.next_if_eq(&'"')?;
        let mut text = String::new();
        loop {
            match chars.next()? {
                '"' => return Some(text),
                '\\' => match chars.next()? {
                    c @ ('"' | '\\' | '/') => text.push(c),
                    'n' => text.push('\n'),
                    'r' => text.push('\r'),
                    't' => text.push('\t'),
                    'b' => text.push('\u{8}'),
                    'f' => text.push('\u{c}'),
                    'u' => {
                        let mut code = Json::hex(chars)?;
                        // A character past the first plane is two, a high one and a low one.
                        if (0xD800..0xDC00).contains(&code) {
                            (chars.next()? == '\\' && chars.next()? == 'u').then_some(())?;
                            code = 0x10000 + ((code - 0xD800) << 10) + (Json::hex(chars)?.checked_sub(0xDC00)?);
                        }
                        text.push(char::from_u32(code)?);
                    }
                    _ => return None,
                },
                c => text.push(c),
            }
        }
    }

    fn hex(chars: &mut Chars) -> Option<u32> {
        (0..4).try_fold(0, |code, _| Some(code * 16 + chars.next()?.to_digit(16)?))
    }

    /// The value of `key`, if this is an object that has it.
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The text this is, if it is a string.
    fn text(&self) -> Option<&str> {
        match self {
            Json::Text(text) => Some(text),
            _ => None,
        }
    }

    /// This as it is written, for an `id` to be given back.
    fn show(&self) -> String {
        match self {
            Json::Number(n) => n.clone(),
            Json::Text(text) => quote(text),
            _ => "null".to_string(),
        }
    }
}

/// What a hover shows at the name of a static, in a kernel that reads it: the doc the static has.
#[test]
fn a_static_named_in_a_kernel_shows_its_doc() -> Result {
    let Some(binary) = rust_analyzer() else {
        eprintln!("no rust-analyzer here: skipped; set RUST_ANALYZER to run it");
        return Ok(());
    };
    let mut editor = Editor::open(&binary, &written("hover_items")?)?;
    let shown = editor.shown("[COUNTS]", 2)?;
    assert!(shown.contains("The counts that `bump` adds one to"), "{shown}");
    Ok(())
}

/// What a hover shows at the name of an instruction, in a module whose kernels compile: its doc, which says what it
/// does, each form it takes with the operands it writes marked, the CPU feature it needs, and where Intel's SDM says
/// the rest; at either word of a name of two; and the doc of the member of a family that is named. A GPU instruction's
/// doc says what each word of its name is, the PTX and the GPU it needs, and where PTX's ISA says the rest; it shows at
/// each word of the name. An instruction only a function that `cfg` leaves out names has none.
#[test]
fn an_instruction_named_in_a_kernel_shows_its_doc() -> Result {
    let Some(binary) = rust_analyzer() else {
        eprintln!("no rust-analyzer here: skipped; set RUST_ANALYZER to run it");
        return Ok(());
    };
    let mut editor = Editor::open(&binary, &written("hover_instructions")?)?;
    let shown = editor.shown("vaddps(", 2)?;
    for want in [
        "Adds the floats of 32 bits of the last two operands, element by element, into the first.",
        "vaddps(=xmm, xmm, xmm/m128)\nvaddps(=ymm, ymm, ymm/m256)",
        "`=` marks an operand it writes. Needs the CPU feature `avx`.",
        "Intel's SDM, volume 2: VADDPS.",
    ] {
        assert!(shown.contains(want), "no `{want}` in the hover at `vaddps`:\n{shown}");
    }
    for skip in [1, 6] {
        let shown = editor.shown("lock.bts(", skip)?;
        assert!(shown.contains("lock.bts(+m64, r64)"), "{shown}");
        assert!(shown.contains("Intel's SDM, volume 2: LOCK, BTS."), "{shown}");
    }
    let shown = editor.shown("cmovz(", 2)?;
    assert!(
        shown.contains("Copies the second operand to the first if zero, that is, if the zero flag is set."),
        "{shown}"
    );
    assert!(shown.contains("cmovz(+r64, r64/m64)"), "{shown}");
    let left_out = editor.hover("src/lib.rs", "vsubps(", 2)?.unwrap_or_default();
    assert!(!left_out.contains("Intel's SDM"), "{left_out}");
    let shown = editor.shown("ld.global.f32(", 1)?;
    for want in [
        "Reads memory at the address in the second operand, and writes it to the first.",
        "* `.global`: global memory, which every thread of the GPU reaches\n* `.f32`: a float of 32 bits",
        "ld.global.f32(=f32/_, [u64])",
        "`=` marks an operand it writes.",
        "PTX's ISA, Data Movement and Conversion Instructions: ld.",
    ] {
        assert!(shown.contains(want), "no `{want}` in the hover at `ld`:\n{shown}");
    }
    for skip in [5, 11] {
        let shown = editor.shown("ld.global.f32(", skip)?;
        assert!(shown.contains("ld.global.f32(=f32/_, [u64])"), "{shown}");
    }
    let shown = editor.shown("fma.rn.ftz.f32(", 1)?;
    assert!(shown.contains("fma.rn.ftz.f32(=f32, f32, f32, f32)"), "{shown}");
    assert!(shown.contains("* `.ftz`: flushes a subnormal input or result to a zero"), "{shown}");
    let shown = editor.shown("shfl.sync.bfly.b32(", 1)?;
    assert!(shown.contains("Needs PTX 6.0 and `sm_30`."), "{shown}");
    let shown = editor.shown("mma.sync.aligned", 1)?;
    let form = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(=(f32/_)*4, %f16x2*4, %f16x2*2, f32*4)";
    assert!(shown.contains(form) && shown.contains("Needs PTX 7.0 and `sm_80`."), "{shown}");
    let left_out = editor.hover("src/lib.rs", "sub.rn.f32(", 1)?.unwrap_or_default();
    assert!(!left_out.contains("PTX's ISA"), "{left_out}");
    Ok(())
}

/// What a hover shows at the name of an instruction where a kernel is refused: its doc, in the kernel that kernite
/// refuses as in the one it compiles; and nothing at a name that x86 or the GPU has no instruction of.
#[test]
fn an_instruction_shows_its_doc_where_a_kernel_is_refused() -> Result {
    let Some(binary) = rust_analyzer() else {
        eprintln!("no rust-analyzer here: skipped; set RUST_ANALYZER to run it");
        return Ok(());
    };
    let mut editor = Editor::open(&binary, &written("hover_instructions_error")?)?;
    for (needle, name) in [("vaddps(", "VADDPS"), ("vsubps(", "VSUBPS")] {
        let shown = editor.shown(needle, 2)?;
        assert!(shown.contains(&format!("Intel's SDM, volume 2: {name}.")), "{shown}");
    }
    let unknown = editor.hover("src/lib.rs", "vaddpz(", 2)?.unwrap_or_default();
    assert!(!unknown.contains("Intel's SDM"), "{unknown}");
    for (needle, name) in [
        ("ld.global.f32(", "Data Movement and Conversion Instructions: ld."),
        ("sub.rn.f32(", "Floating-Point Instructions: sub."),
    ] {
        let shown = editor.shown(needle, 1)?;
        assert!(shown.contains(&format!("PTX's ISA, {name}")), "{shown}");
    }
    let unknown = editor.hover("src/lib.rs", "ld.globl.f32(", 1)?.unwrap_or_default();
    assert!(!unknown.contains("PTX's ISA"), "{unknown}");
    Ok(())
}
