//! Evaluates the integer `#define`s of `sid.h`: object-like constants and one-parameter field
//! macros such as `(((unsigned)(x) & 0xf) << 8)`, with earlier macros in scope.

use std::collections::HashMap;

/// A macro as `sid.h` defines it.
#[derive(Clone, Debug)]
pub enum Macro {
    Object(String),
    Function { parameters: Vec<String>, body: String },
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(u64),
    Name(String),
    Punct(&'static str),
}

const PUNCTS: [&str; 18] = ["<<", ">>", "(", ")", "~", "!", "*", "/", "%", "+", "-", "&", "^", "|", "?", ":", ",", "#"];

fn tokens(text: &str) -> Option<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut at = 0;
    let mut out = Vec::new();
    while at < bytes.len() {
        let byte = bytes[at];
        if byte.is_ascii_whitespace() {
            at += 1;
        } else if byte.is_ascii_digit() {
            let start = at;
            while at < bytes.len() && (bytes[at].is_ascii_alphanumeric()) {
                at += 1;
            }
            let literal = text[start..at].trim_end_matches(['u', 'U', 'l', 'L']);
            let value = if let Some(hex) = literal.strip_prefix("0x").or_else(|| literal.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).ok()?
            } else {
                literal.parse().ok()?
            };
            out.push(Token::Number(value));
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = at;
            while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                at += 1;
            }
            out.push(Token::Name(text[start..at].to_string()));
        } else {
            let punct = PUNCTS.iter().find(|punct| text[at..].starts_with(**punct))?;
            out.push(Token::Punct(punct));
            at += punct.len();
        }
    }
    Some(out)
}

/// Evaluates expressions over the macros defined so far.
pub struct Evaluator<'a> {
    pub macros: &'a HashMap<String, Macro>,
}

const CASTS: [&str; 6] = ["unsigned", "uint32_t", "uint64_t", "int", "long", "uint"];
const DEPTH_LIMIT: u32 = 32;

impl Evaluator<'_> {
    /// The value of `text` with `bindings` for macro parameters; `None` when it is not a plain
    /// integer expression.
    pub fn eval(&self, text: &str, bindings: &[(String, u64)]) -> Option<u64> {
        self.eval_depth(text, bindings, 0)
    }

    fn eval_depth(&self, text: &str, bindings: &[(String, u64)], depth: u32) -> Option<u64> {
        if depth > DEPTH_LIMIT {
            return None;
        }
        let tokens = tokens(text)?;
        let mut parser = Parser { tokens: &tokens, at: 0, eval: self, bindings, depth };
        let value = parser.expression(0)?;
        (parser.at == tokens.len()).then_some(value)
    }
}

struct Parser<'a, 'b> {
    tokens: &'a [Token],
    at: usize,
    eval: &'a Evaluator<'b>,
    bindings: &'a [(String, u64)],
    depth: u32,
}

fn precedence(punct: &str) -> Option<(u32, fn(u64, u64) -> Option<u64>)> {
    Some(match punct {
        "*" => (10, |a, b| Some(a.wrapping_mul(b))),
        "/" => (10, |a, b| a.checked_div(b)),
        "%" => (10, |a, b| a.checked_rem(b)),
        "+" => (9, |a, b| Some(a.wrapping_add(b))),
        "-" => (9, |a, b| Some(a.wrapping_sub(b))),
        "<<" => (8, |a, b| (b < 64).then(|| a << b)),
        ">>" => (8, |a, b| (b < 64).then(|| a >> b)),
        "&" => (7, |a, b| Some(a & b)),
        "^" => (6, |a, b| Some(a ^ b)),
        "|" => (5, |a, b| Some(a | b)),
        _ => return None,
    })
}

impl Parser<'_, '_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn punct(&mut self, punct: &str) -> bool {
        if self.peek() == Some(&Token::Punct(match punct {
            "(" => "(",
            ")" => ")",
            "," => ",",
            "?" => "?",
            ":" => ":",
            _ => return false,
        })) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expression(&mut self, minimum: u32) -> Option<u64> {
        let mut left = self.unary()?;
        loop {
            if minimum == 0 && self.punct("?") {
                let then = self.expression(0)?;
                if !self.punct(":") {
                    return None;
                }
                let otherwise = self.expression(0)?;
                left = if left != 0 { then } else { otherwise };
                continue;
            }
            let Some(Token::Punct(punct)) = self.peek() else { break };
            let Some((level, apply)) = precedence(punct) else { break };
            if level < minimum.max(1) {
                break;
            }
            self.at += 1;
            let right = self.expression(level + 1)?;
            left = apply(left, right)?;
        }
        Some(left)
    }

    fn unary(&mut self) -> Option<u64> {
        match self.peek()?.clone() {
            Token::Punct("~") => {
                self.at += 1;
                Some(!self.unary()?)
            }
            Token::Punct("-") => {
                self.at += 1;
                Some(self.unary()?.wrapping_neg())
            }
            Token::Punct("!") => {
                self.at += 1;
                Some(u64::from(self.unary()? == 0))
            }
            Token::Punct("(") => {
                // A cast such as `(unsigned)` or a parenthesised expression.
                if let (Some(Token::Name(name)), Some(Token::Punct(")"))) = (self.tokens.get(self.at + 1), self.tokens.get(self.at + 2)) {
                    if CASTS.contains(&name.as_str()) {
                        self.at += 3;
                        return self.unary();
                    }
                }
                self.at += 1;
                let value = self.expression(0)?;
                self.punct(")").then_some(value)
            }
            Token::Number(value) => {
                self.at += 1;
                Some(value)
            }
            Token::Name(name) => {
                self.at += 1;
                if let Some(&(_, value)) = self.bindings.iter().find(|(parameter, _)| *parameter == name) {
                    return Some(value);
                }
                match self.eval.macros.get(&name)? {
                    Macro::Object(body) => self.eval.eval_depth(body, &[], self.depth + 1),
                    Macro::Function { parameters, body } => {
                        if !self.punct("(") {
                            return None;
                        }
                        let mut arguments = Vec::new();
                        if !self.punct(")") {
                            loop {
                                arguments.push(self.expression(0)?);
                                if self.punct(")") {
                                    break;
                                }
                                if !self.punct(",") {
                                    return None;
                                }
                            }
                        }
                        if arguments.len() != parameters.len() {
                            return None;
                        }
                        let bindings: Vec<(String, u64)> = parameters.iter().cloned().zip(arguments).collect();
                        self.eval.eval_depth(body, &bindings, self.depth + 1)
                    }
                }
            }
            Token::Punct(_) => None,
        }
    }
}

/// Every `#define` of a C header, in order, with continuation lines joined; whether each was
/// indented (`#define   NAME`, the style `sid.h` uses for a packet's fields).
pub fn defines(header: &str) -> Vec<(String, Macro, bool)> {
    let mut joined = Vec::new();
    let mut current = String::new();
    for line in header.lines() {
        let line = line.trim_end();
        if let Some(stripped) = line.strip_suffix('\\') {
            current.push_str(stripped);
            current.push(' ');
            continue;
        }
        current.push_str(line);
        joined.push(std::mem::take(&mut current));
    }
    let mut out = Vec::new();
    for line in joined {
        let Some(rest) = line.trim_start().strip_prefix("#define") else { continue };
        let indented = rest.starts_with("  ");
        let rest = rest.trim_start();
        let name_end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
        let name = rest[..name_end].to_string();
        if name.is_empty() {
            continue;
        }
        let after = &rest[name_end..];
        let strip_comment = |text: &str| -> String {
            let text = match text.find("/*") {
                Some(at) => &text[..at],
                None => text,
            };
            match text.find("//") {
                Some(at) => text[..at].trim().to_string(),
                None => text.trim().to_string(),
            }
        };
        let definition = if let Some(arguments) = after.strip_prefix('(') {
            let Some(close) = arguments.find(')') else { continue };
            let parameters = arguments[..close].split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
            Macro::Function { parameters, body: strip_comment(&arguments[close + 1..]) }
        } else {
            Macro::Object(strip_comment(after))
        };
        out.push((name, definition, indented));
    }
    out
}
