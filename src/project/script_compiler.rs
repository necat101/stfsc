//! Compile editable FuckScript source into native Rust script implementations.
//!
//! This is intentionally a small, predictable compiler. It accepts the editor
//! syntax used in `ruleset_scripting.txt` and emits Rust that implements the
//! runtime `FuckScript` trait. There is no interpreter in the game loop.

use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct NativeScriptSource {
    pub runtime_name: String,
    pub struct_name: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptDiagnosticLevel {
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptDiagnostic {
    pub level: ScriptDiagnosticLevel,
    pub message: String,
}

impl ScriptDiagnostic {
    fn error(message: impl Into<String>) -> Self {
        Self {
            level: ScriptDiagnosticLevel::Error,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct ScriptAst {
    name: String,
    hooks: Vec<HookAst>,
}

#[derive(Clone, Debug)]
struct HookAst {
    name: String,
    body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
enum Stmt {
    Let {
        name: String,
        expr: Expr,
    },
    Assign {
        target: AssignTarget,
        expr: Expr,
    },
    Expr(Expr),
    If {
        condition: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Vec<Stmt>,
    },
}

#[derive(Clone, Debug)]
enum AssignTarget {
    State(String),
    Transform(Vec<String>),
}

#[derive(Clone, Debug)]
enum Expr {
    Number(String),
    Bool(bool),
    String(String),
    Path(Vec<String>),
    Call {
        callee: Vec<String>,
        args: Vec<Expr>,
    },
    Unary {
        op: String,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: String,
        right: Box<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    F32,
    Bool,
    String,
    Vec3,
    Quat,
    U32,
    Entity,
    Event,
    XrSnapshot,
    Unknown,
}

impl ValueType {
    fn rust_type(self) -> &'static str {
        match self {
            ValueType::F32 => "f32",
            ValueType::Bool => "bool",
            ValueType::String => "String",
            ValueType::Vec3 => "glam::Vec3",
            ValueType::Quat => "glam::Quat",
            ValueType::U32 => "u32",
            ValueType::Entity => "hecs::Entity",
            ValueType::Event | ValueType::XrSnapshot | ValueType::Unknown => "()",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Number(String),
    String(String),
    Symbol(char),
    Operator(String),
    Eof,
}

struct Lexer<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().peekable(),
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, ScriptDiagnostic> {
        let mut tokens = Vec::new();
        while let Some(ch) = self.chars.peek().copied() {
            match ch {
                c if c.is_whitespace() => {
                    self.chars.next();
                }
                '/' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'/') {
                        while let Some(c) = self.chars.next() {
                            if c == '\n' {
                                break;
                            }
                        }
                    } else {
                        tokens.push(Token::Operator("/".to_string()));
                    }
                }
                '"' => tokens.push(self.lex_string()?),
                '0'..='9' => tokens.push(self.lex_number()),
                'a'..='z' | 'A'..='Z' | '_' => tokens.push(self.lex_ident()),
                '{' | '}' | '(' | ')' | ',' | ';' | '.' => {
                    self.chars.next();
                    tokens.push(Token::Symbol(ch));
                }
                '+' | '-' | '*' | '=' | '!' | '<' | '>' | '&' | '|' => {
                    tokens.push(self.lex_operator()?);
                }
                _ => {
                    return Err(ScriptDiagnostic::error(format!(
                        "Unexpected character '{}'",
                        ch
                    )));
                }
            }
        }
        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn lex_ident(&mut self) -> Token {
        let mut text = String::new();
        while let Some(ch) = self.chars.peek().copied() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                text.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        Token::Ident(text)
    }

    fn lex_number(&mut self) -> Token {
        let mut text = String::new();
        while let Some(ch) = self.chars.peek().copied() {
            if ch.is_ascii_digit() || ch == '.' {
                text.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        Token::Number(text)
    }

    fn lex_string(&mut self) -> Result<Token, ScriptDiagnostic> {
        self.chars.next();
        let mut text = String::new();
        while let Some(ch) = self.chars.next() {
            match ch {
                '"' => return Ok(Token::String(text)),
                '\\' => {
                    let escaped = self.chars.next().ok_or_else(|| {
                        ScriptDiagnostic::error("Unterminated escape sequence in string")
                    })?;
                    match escaped {
                        '"' => text.push('"'),
                        '\\' => text.push('\\'),
                        'n' => text.push('\n'),
                        't' => text.push('\t'),
                        other => text.push(other),
                    }
                }
                other => text.push(other),
            }
        }
        Err(ScriptDiagnostic::error("Unterminated string literal"))
    }

    fn lex_operator(&mut self) -> Result<Token, ScriptDiagnostic> {
        let first = self.chars.next().unwrap();
        let mut op = String::new();
        op.push(first);

        if let Some(next) = self.chars.peek().copied() {
            let two = matches!(
                (first, next),
                ('=', '=') | ('!', '=') | ('<', '=') | ('>', '=') | ('&', '&') | ('|', '|')
            );
            if two {
                op.push(next);
                self.chars.next();
            } else if first == '&' || first == '|' {
                return Err(ScriptDiagnostic::error(format!(
                    "Use '{}' twice for logical operators",
                    first
                )));
            }
        }

        Ok(Token::Operator(op))
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn parse_script(&mut self) -> Result<ScriptAst, ScriptDiagnostic> {
        self.expect_ident_text("script")?;
        let name = self.expect_ident()?;
        self.expect_symbol('{')?;

        let mut hooks = Vec::new();
        while !self.consume_symbol('}') {
            if self.peek_is_eof() {
                return Err(ScriptDiagnostic::error("Expected '}' to close script"));
            }
            hooks.push(self.parse_hook()?);
        }

        Ok(ScriptAst { name, hooks })
    }

    fn parse_hook(&mut self) -> Result<HookAst, ScriptDiagnostic> {
        let name = self.expect_ident()?;
        self.expect_symbol('(')?;
        while !self.consume_symbol(')') {
            if self.peek_is_eof() {
                return Err(ScriptDiagnostic::error(format!(
                    "Expected ')' in hook '{}'",
                    name
                )));
            }
            self.pos += 1;
        }
        let body = self.parse_block()?;
        Ok(HookAst { name, body })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ScriptDiagnostic> {
        self.expect_symbol('{')?;
        let mut body = Vec::new();
        while !self.consume_symbol('}') {
            if self.peek_is_eof() {
                return Err(ScriptDiagnostic::error("Expected '}' to close block"));
            }
            body.push(self.parse_stmt()?);
        }
        Ok(body)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ScriptDiagnostic> {
        if self.consume_ident_text("let") {
            let name = self.expect_ident()?;
            self.expect_operator("=")?;
            let expr = self.parse_expr(0)?;
            self.expect_symbol(';')?;
            return Ok(Stmt::Let { name, expr });
        }

        if self.consume_ident_text("if") {
            let condition = self.parse_expr(0)?;
            let then_branch = self.parse_block()?;
            let else_branch = if self.consume_ident_text("else") {
                self.parse_block()?
            } else {
                Vec::new()
            };
            return Ok(Stmt::If {
                condition,
                then_branch,
                else_branch,
            });
        }

        let expr = self.parse_expr(0)?;
        if self.consume_operator("=") {
            let target = match expr {
                Expr::Path(path) => Self::assign_target_from_path(path)?,
                _ => {
                    return Err(ScriptDiagnostic::error(
                        "Assignment target must be ctx.state.* or ctx.transform.*",
                    ));
                }
            };
            let expr = self.parse_expr(0)?;
            self.expect_symbol(';')?;
            return Ok(Stmt::Assign { target, expr });
        }

        self.expect_symbol(';')?;
        Ok(Stmt::Expr(expr))
    }

    fn assign_target_from_path(path: Vec<String>) -> Result<AssignTarget, ScriptDiagnostic> {
        if path.len() == 3 && path[0] == "ctx" && path[1] == "state" {
            return Ok(AssignTarget::State(path[2].clone()));
        }

        if path.len() >= 3 && path[0] == "ctx" && path[1] == "transform" {
            let tail = path[2..].to_vec();
            if matches!(
                tail.first().map(String::as_str),
                Some("position" | "rotation" | "scale")
            ) {
                return Ok(AssignTarget::Transform(tail));
            }
        }

        Err(ScriptDiagnostic::error(
            "Assignment target must be ctx.state.* or ctx.transform.position/rotation/scale",
        ))
    }

    fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, ScriptDiagnostic> {
        let mut left = self.parse_unary()?;

        loop {
            let Some(op) = self.peek_operator_string() else {
                break;
            };
            let Some(prec) = precedence(&op) else {
                break;
            };
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let right = self.parse_expr(prec + 1)?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ScriptDiagnostic> {
        if let Some(op) = self.peek_operator_string() {
            if op == "!" || op == "-" {
                self.pos += 1;
                return Ok(Expr::Unary {
                    op,
                    expr: Box::new(self.parse_unary()?),
                });
            }
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, ScriptDiagnostic> {
        match self.next().clone() {
            Token::Number(value) => Ok(Expr::Number(value)),
            Token::String(value) => Ok(Expr::String(value)),
            Token::Ident(value) if value == "true" => Ok(Expr::Bool(true)),
            Token::Ident(value) if value == "false" => Ok(Expr::Bool(false)),
            Token::Ident(value) => {
                let mut path = vec![value];
                while self.consume_symbol('.') {
                    path.push(self.expect_ident()?);
                }

                if self.consume_symbol('(') {
                    let mut args = Vec::new();
                    if !self.consume_symbol(')') {
                        loop {
                            args.push(self.parse_expr(0)?);
                            if self.consume_symbol(')') {
                                break;
                            }
                            self.expect_symbol(',')?;
                        }
                    }
                    Ok(Expr::Call { callee: path, args })
                } else {
                    Ok(Expr::Path(path))
                }
            }
            Token::Symbol('(') => {
                let expr = self.parse_expr(0)?;
                self.expect_symbol(')')?;
                Ok(expr)
            }
            other => Err(ScriptDiagnostic::error(format!(
                "Expected expression, found {:?}",
                other
            ))),
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn next(&mut self) -> &Token {
        let pos = self.pos;
        self.pos += 1;
        self.tokens.get(pos).unwrap_or(&Token::Eof)
    }

    fn peek_is_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn expect_ident(&mut self) -> Result<String, ScriptDiagnostic> {
        match self.next().clone() {
            Token::Ident(value) => Ok(value),
            other => Err(ScriptDiagnostic::error(format!(
                "Expected identifier, found {:?}",
                other
            ))),
        }
    }

    fn expect_ident_text(&mut self, expected: &str) -> Result<(), ScriptDiagnostic> {
        let found = self.expect_ident()?;
        if found == expected {
            Ok(())
        } else {
            Err(ScriptDiagnostic::error(format!(
                "Expected '{}', found '{}'",
                expected, found
            )))
        }
    }

    fn consume_ident_text(&mut self, expected: &str) -> bool {
        match self.peek() {
            Token::Ident(found) if found == expected => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    fn expect_symbol(&mut self, expected: char) -> Result<(), ScriptDiagnostic> {
        if self.consume_symbol(expected) {
            Ok(())
        } else {
            Err(ScriptDiagnostic::error(format!(
                "Expected '{}', found {:?}",
                expected,
                self.peek()
            )))
        }
    }

    fn consume_symbol(&mut self, expected: char) -> bool {
        match self.peek() {
            Token::Symbol(found) if *found == expected => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    fn expect_operator(&mut self, expected: &str) -> Result<(), ScriptDiagnostic> {
        if self.consume_operator(expected) {
            Ok(())
        } else {
            Err(ScriptDiagnostic::error(format!(
                "Expected '{}', found {:?}",
                expected,
                self.peek()
            )))
        }
    }

    fn consume_operator(&mut self, expected: &str) -> bool {
        match self.peek() {
            Token::Operator(found) if found == expected => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    fn peek_operator_string(&self) -> Option<String> {
        match self.peek() {
            Token::Operator(op) => Some(op.clone()),
            _ => None,
        }
    }
}

fn precedence(op: &str) -> Option<u8> {
    match op {
        "||" => Some(1),
        "&&" => Some(2),
        "==" | "!=" => Some(3),
        "<" | "<=" | ">" | ">=" => Some(4),
        "+" | "-" => Some(5),
        "*" | "/" => Some(6),
        _ => None,
    }
}

pub fn render_native_script_module(
    scripts: &[NativeScriptSource],
) -> Result<String, Vec<ScriptDiagnostic>> {
    let mut diagnostics = Vec::new();
    let mut compiled = Vec::new();

    for script in scripts {
        match compile_one(script) {
            Ok(source) => compiled.push(source),
            Err(mut errors) => diagnostics.append(&mut errors),
        }
    }

    if diagnostics
        .iter()
        .any(|diag| diag.level == ScriptDiagnosticLevel::Error)
    {
        return Err(diagnostics);
    }

    let mut out = String::new();
    out.push_str("// Generated by STFSC export. Do not edit by hand.\n");
    out.push_str("use super::scripting::{FuckScript, ScriptContext, ScriptFurrySpecies, ScriptGameMode, ScriptLightType, ScriptPrimitive, ScriptRegistry, ScriptResourceKind, ScriptSandboxProfile, XrHand, XrPoseSpace};\n\n");

    if compiled.is_empty() {
        out.push_str("pub fn register_generated_scripts(_registry: &mut ScriptRegistry) {}\n");
        return Ok(out);
    }

    for source in &compiled {
        out.push_str(source);
        out.push('\n');
    }

    out.push_str("pub fn register_generated_scripts(registry: &mut ScriptRegistry) {\n");
    for script in scripts {
        out.push_str(&format!(
            "    registry.register({:?}, || {}::default());\n",
            script.runtime_name, script.struct_name
        ));
    }
    out.push_str("}\n");
    Ok(out)
}

fn compile_one(script: &NativeScriptSource) -> Result<String, Vec<ScriptDiagnostic>> {
    let tokens = match Lexer::new(&script.source).tokenize() {
        Ok(tokens) => tokens,
        Err(err) => return Err(vec![err]),
    };

    let ast = match Parser::new(tokens).parse_script() {
        Ok(ast) => ast,
        Err(err) => return Err(vec![err]),
    };

    let mut diagnostics = Vec::new();
    if ast.name.trim().is_empty() {
        diagnostics.push(ScriptDiagnostic::error("Script name cannot be empty"));
    }

    for hook in &ast.hooks {
        if hook_spec(&hook.name).is_none() {
            diagnostics.push(ScriptDiagnostic::error(format!(
                "Unsupported hook '{}'",
                hook.name
            )));
        }
    }

    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let mut state_types = BTreeMap::new();
    for hook in &ast.hooks {
        collect_state_types(&hook.body, &mut state_types);
    }

    let mut out = String::new();
    out.push_str("#[derive(Default)]\n");
    out.push_str(&format!("pub struct {} {{\n", script.struct_name));
    for (name, ty) in &state_types {
        out.push_str(&format!(
            "    {}: {},\n",
            state_field_ident(name),
            ty.rust_type()
        ));
    }
    out.push_str("}\n\n");

    out.push_str(&format!("impl FuckScript for {} {{\n", script.struct_name));
    for hook in &ast.hooks {
        let spec = hook_spec(&hook.name).expect("hook spec was already checked");
        out.push_str("    ");
        out.push_str(spec.signature);
        out.push_str(" {\n");

        let mut codegen = Codegen::new(&state_types);
        codegen.seed_hook_env(spec);
        for stmt in &hook.body {
            match codegen.write_stmt(stmt, 2) {
                Ok(()) => {}
                Err(err) => diagnostics.push(err),
            }
        }
        out.push_str(&codegen.out);
        out.push_str("    }\n");
    }
    out.push_str("}\n");

    if diagnostics.is_empty() {
        Ok(out)
    } else {
        Err(diagnostics)
    }
}

fn collect_state_types(stmts: &[Stmt], state_types: &mut BTreeMap<String, ValueType>) {
    let env = HashMap::new();
    for stmt in stmts {
        match stmt {
            Stmt::Assign {
                target: AssignTarget::State(name),
                expr,
            } => {
                let ty = infer_expr_type(expr, &env, state_types);
                let ty = if ty == ValueType::Unknown {
                    ValueType::F32
                } else {
                    ty
                };
                state_types.entry(name.clone()).or_insert(ty);
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_state_types(then_branch, state_types);
                collect_state_types(else_branch, state_types);
            }
            _ => {}
        }
    }
}

fn infer_expr_type(
    expr: &Expr,
    env: &HashMap<String, ValueType>,
    state_types: &BTreeMap<String, ValueType>,
) -> ValueType {
    match expr {
        Expr::Number(_) => ValueType::F32,
        Expr::Bool(_) => ValueType::Bool,
        Expr::String(_) => ValueType::String,
        Expr::Path(path) => infer_path_type(path, env, state_types),
        Expr::Call { callee, args } => infer_call_type(callee, args, env, state_types),
        Expr::Unary { op, expr } => {
            if op == "!" {
                ValueType::Bool
            } else {
                infer_expr_type(expr, env, state_types)
            }
        }
        Expr::Binary { op, left, .. } => match op.as_str() {
            "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" => ValueType::Bool,
            _ => infer_expr_type(left, env, state_types),
        },
    }
}

fn infer_path_type(
    path: &[String],
    env: &HashMap<String, ValueType>,
    state_types: &BTreeMap<String, ValueType>,
) -> ValueType {
    if path.is_empty() {
        return ValueType::Unknown;
    }

    if path.len() == 1 {
        return env.get(&path[0]).copied().unwrap_or(ValueType::Unknown);
    }

    if path[0] == "ctx" {
        if path == ["ctx", "dt"] {
            return ValueType::F32;
        }
        if path.len() == 3 && path[1] == "state" {
            return state_types.get(&path[2]).copied().unwrap_or(ValueType::F32);
        }
        if path.len() >= 3 && path[1] == "transform" {
            return infer_transform_path_type(&path[2..]);
        }
    }

    match env.get(&path[0]).copied() {
        Some(ValueType::Vec3) if path.len() == 2 && matches_xyz(&path[1]) => ValueType::F32,
        Some(ValueType::Quat) if path.len() == 2 && matches_xyzw(&path[1]) => ValueType::F32,
        Some(ty) => ty,
        None => ValueType::Unknown,
    }
}

fn infer_transform_path_type(path: &[String]) -> ValueType {
    match path {
        [field] if field == "position" || field == "scale" => ValueType::Vec3,
        [field] if field == "rotation" => ValueType::Quat,
        [field, axis] if (field == "position" || field == "scale") && matches_xyz(axis) => {
            ValueType::F32
        }
        [field, axis] if field == "rotation" && matches_xyzw(axis) => ValueType::F32,
        _ => ValueType::Unknown,
    }
}

fn infer_call_type(
    callee: &[String],
    args: &[Expr],
    env: &HashMap<String, ValueType>,
    state_types: &BTreeMap<String, ValueType>,
) -> ValueType {
    match callee {
        [name] if name == "vec3" => ValueType::Vec3,
        [name] if name == "quat_identity" || name == "quat_from_rotation_y" => ValueType::Quat,
        [name] if name == "sin" || name == "atan2" || name == "length" => ValueType::F32,
        [name] if name == "length_squared" || name == "clamp" => ValueType::F32,
        [name] if name == "slerp" => args
            .first()
            .map(|expr| infer_expr_type(expr, env, state_types))
            .unwrap_or(ValueType::Unknown),
        [ctx, helper] if ctx == "ctx" && helper == "player_position" => ValueType::Vec3,
        [ctx, helper] if ctx == "ctx" && helper == "distance_to_player" => ValueType::F32,
        [ctx, helper] if ctx == "ctx" && helper == "has_player" => ValueType::Bool,
        [ctx, helper] if ctx == "ctx" && helper == "track_player" => ValueType::Bool,
        [ctx, helper]
            if ctx == "ctx"
                && matches!(
                    helper.as_str(),
                    "spawn_primitive"
                        | "spawn_ground"
                        | "spawn_light"
                        | "spawn_light_with_cones"
                        | "spawn_scatter"
                        | "spawn_scatter_range"
                        | "spawn_grid"
                        | "spawn_resource_cluster"
                        | "spawn_tree_cluster"
                        | "spawn_campfire"
                        | "spawn_furry_npc"
                ) =>
        {
            ValueType::U32
        }
        [ctx, helper]
            if ctx == "ctx"
                && matches!(
                    helper.as_str(),
                    "action_pressed" | "action_just_pressed" | "action_just_released"
                ) =>
        {
            ValueType::Bool
        }
        [ctx, helper] if ctx == "ctx" && helper == "action_value" => ValueType::F32,
        [ctx, helper]
            if ctx == "ctx"
                && matches!(
                    helper.as_str(),
                    "log"
                        | "despawn_self"
                        | "request_haptic"
                        | "track_transform_to_xr_pose"
                        | "fire_projectile"
                        | "fire_projectile_from_self"
                        | "set_procedural_generation"
                        | "set_game_mode"
                        | "set_sandbox_clock"
                        | "configure_sandbox"
                        | "configure_twilight_survival"
                        | "set_script"
                        | "set_scripts"
                ) =>
        {
            ValueType::Unknown
        }
        _ => ValueType::Unknown,
    }
}

struct HookSpec {
    signature: &'static str,
    locals: &'static [(&'static str, ValueType)],
}

fn hook_spec(name: &str) -> Option<HookSpec> {
    match name {
        "on_awake" => Some(HookSpec {
            signature: "fn on_awake(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_start" => Some(HookSpec {
            signature: "fn on_start(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_update" => Some(HookSpec {
            signature: "fn on_update(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_fixed_update" => Some(HookSpec {
            signature: "fn on_fixed_update(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_late_update" => Some(HookSpec {
            signature: "fn on_late_update(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_enable" => Some(HookSpec {
            signature: "fn on_enable(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_disable" => Some(HookSpec {
            signature: "fn on_disable(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_destroy" => Some(HookSpec {
            signature: "fn on_destroy(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_collision_start" => Some(HookSpec {
            signature: "fn on_collision_start(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_collision_stay" => Some(HookSpec {
            signature: "fn on_collision_stay(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_collision_end" => Some(HookSpec {
            signature: "fn on_collision_end(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_trigger_start" => Some(HookSpec {
            signature: "fn on_trigger_start(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_trigger_stay" => Some(HookSpec {
            signature: "fn on_trigger_stay(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_trigger_end" => Some(HookSpec {
            signature: "fn on_trigger_end(&mut self, ctx: &mut ScriptContext, other: hecs::Entity)",
            locals: &[("other", ValueType::Entity)],
        }),
        "on_player_respawn" => Some(HookSpec {
            signature: "fn on_player_respawn(&mut self, ctx: &mut ScriptContext)",
            locals: &[],
        }),
        "on_ui_event" => Some(HookSpec {
            signature:
                "fn on_ui_event(&mut self, ctx: &mut ScriptContext, event: &crate::ui::UiEvent)",
            locals: &[("event", ValueType::Event)],
        }),
        "on_animation_event" => Some(HookSpec {
            signature: "fn on_animation_event(&mut self, ctx: &mut ScriptContext, event_name: &str)",
            locals: &[("event_name", ValueType::String)],
        }),
        "on_xr_frame" => Some(HookSpec {
            signature: "fn on_xr_frame(&mut self, ctx: &mut ScriptContext, xr: &super::scripting::XrInputSnapshot)",
            locals: &[("xr", ValueType::XrSnapshot)],
        }),
        "on_xr_input" => Some(HookSpec {
            signature: "fn on_xr_input(&mut self, ctx: &mut ScriptContext, event: &super::scripting::XrInputEvent)",
            locals: &[("event", ValueType::Event)],
        }),
        "on_application_pause" => Some(HookSpec {
            signature: "fn on_application_pause(&mut self, ctx: &mut ScriptContext, paused: bool)",
            locals: &[("paused", ValueType::Bool)],
        }),
        "on_application_focus" => Some(HookSpec {
            signature: "fn on_application_focus(&mut self, ctx: &mut ScriptContext, focused: bool)",
            locals: &[("focused", ValueType::Bool)],
        }),
        _ => None,
    }
}

struct Codegen<'a> {
    state_types: &'a BTreeMap<String, ValueType>,
    locals: HashMap<String, ValueType>,
    out: String,
    temp_index: usize,
}

impl<'a> Codegen<'a> {
    fn new(state_types: &'a BTreeMap<String, ValueType>) -> Self {
        Self {
            state_types,
            locals: HashMap::new(),
            out: String::new(),
            temp_index: 0,
        }
    }

    fn seed_hook_env(&mut self, spec: HookSpec) {
        for (name, ty) in spec.locals {
            self.locals.insert((*name).to_string(), *ty);
        }
    }

    fn write_stmt(&mut self, stmt: &Stmt, indent: usize) -> Result<(), ScriptDiagnostic> {
        match stmt {
            Stmt::Let { name, expr } => {
                let rust = self.expr_to_rust(expr)?;
                let ty = infer_expr_type(expr, &self.locals, self.state_types);
                self.locals.insert(name.clone(), ty);
                self.line(indent, &format!("let {} = {};", rust_ident(name), rust));
            }
            Stmt::Assign { target, expr } => match target {
                AssignTarget::State(name) => {
                    let rust = self.expr_to_rust(expr)?;
                    self.line(
                        indent,
                        &format!("self.{} = {};", state_field_ident(name), rust),
                    );
                }
                AssignTarget::Transform(path) => {
                    let rust = self.expr_to_rust(expr)?;
                    let temp = self.temp_name();
                    self.line(indent, &format!("let {} = {};", temp, rust));
                    self.line(indent, "if let Some(mut transform) = ctx.transform_mut() {");
                    self.line(
                        indent + 1,
                        &format!("transform.{} = {};", path.join("."), temp),
                    );
                    self.line(indent, "}");
                }
            },
            Stmt::Expr(expr) => {
                if let Expr::Call { callee, args } = expr {
                    if self.write_context_call(callee, args, indent)? {
                        return Ok(());
                    }
                }
                let rust = self.expr_to_rust(expr)?;
                self.line(indent, &format!("let _ = {};", rust));
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = self.expr_to_rust(condition)?;
                self.line(indent, &format!("if {} {{", condition));
                for stmt in then_branch {
                    self.write_stmt(stmt, indent + 1)?;
                }
                if else_branch.is_empty() {
                    self.line(indent, "}");
                } else {
                    self.line(indent, "} else {");
                    for stmt in else_branch {
                        self.write_stmt(stmt, indent + 1)?;
                    }
                    self.line(indent, "}");
                }
            }
        }

        Ok(())
    }

    fn write_context_call(
        &mut self,
        callee: &[String],
        args: &[Expr],
        indent: usize,
    ) -> Result<bool, ScriptDiagnostic> {
        match callee {
            [ctx, helper] if ctx == "ctx" && helper == "log" => {
                self.expect_arg_count(helper, args, 1)?;
                let message = self.expr_to_rust(&args[0])?;
                self.line(indent, &format!("ctx.log({});", message));
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "despawn_self" => {
                self.expect_arg_count(helper, args, 0)?;
                self.line(indent, "ctx.despawn_self();");
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "request_haptic" => {
                self.expect_arg_count(helper, args, 3)?;
                let hand = hand_arg_to_rust(&args[0])?;
                let amplitude = self.expr_to_rust(&args[1])?;
                let seconds = self.expr_to_rust(&args[2])?;
                self.line(
                    indent,
                    &format!("ctx.request_haptic({}, {}, {});", hand, amplitude, seconds),
                );
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "track_transform_to_xr_pose" => {
                self.expect_arg_count(helper, args, 3)?;
                let pose = pose_arg_to_rust(&args[0])?;
                let position = self.expr_to_rust(&args[1])?;
                let rotation = self.expr_to_rust(&args[2])?;
                self.line(
                    indent,
                    &format!(
                        "ctx.track_transform_to_xr_pose({}, {}, {});",
                        pose, position, rotation
                    ),
                );
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "track_player" => {
                self.expect_arg_count(helper, args, 4)?;
                let radius = self.expr_to_rust(&args[0])?;
                let speed = self.expr_to_rust(&args[1])?;
                let stop_distance = self.expr_to_rust(&args[2])?;
                let turn_speed = self.expr_to_rust(&args[3])?;
                self.line(
                    indent,
                    &format!(
                        "let _ = ctx.track_player({}, {}, {}, {});",
                        radius, speed, stop_distance, turn_speed
                    ),
                );
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "fire_projectile" => {
                self.expect_arg_count(helper, args, 7)?;
                let origin = self.expr_to_rust(&args[0])?;
                let direction = self.expr_to_rust(&args[1])?;
                let speed = self.expr_to_rust(&args[2])?;
                let lifetime = self.expr_to_rust(&args[3])?;
                let radius = self.expr_to_rust(&args[4])?;
                let damage = self.expr_to_rust(&args[5])?;
                let gravity_scale = self.expr_to_rust(&args[6])?;
                self.line(
                    indent,
                    &format!(
                        "ctx.fire_projectile({}, {}, {}, {}, {}, {}, {});",
                        origin, direction, speed, lifetime, radius, damage, gravity_scale
                    ),
                );
                Ok(true)
            }
            [ctx, helper] if ctx == "ctx" && helper == "fire_projectile_from_self" => {
                self.expect_arg_count(helper, args, 5)?;
                let speed = self.expr_to_rust(&args[0])?;
                let lifetime = self.expr_to_rust(&args[1])?;
                let radius = self.expr_to_rust(&args[2])?;
                let damage = self.expr_to_rust(&args[3])?;
                let gravity_scale = self.expr_to_rust(&args[4])?;
                self.line(
                    indent,
                    &format!(
                        "ctx.fire_projectile_from_self({}, {}, {}, {}, {});",
                        speed, lifetime, radius, damage, gravity_scale
                    ),
                );
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn expr_to_rust(&mut self, expr: &Expr) -> Result<String, ScriptDiagnostic> {
        match expr {
            Expr::Number(value) => Ok(number_to_f32(value)),
            Expr::Bool(value) => Ok(value.to_string()),
            Expr::String(value) => Ok(format!("{:?}", value)),
            Expr::Path(path) => self.path_to_rust(path),
            Expr::Call { callee, args } => self.call_to_rust(callee, args),
            Expr::Unary { op, expr } => Ok(format!("({}{})", op, self.expr_to_rust(expr)?)),
            Expr::Binary { left, op, right } => Ok(format!(
                "({} {} {})",
                self.expr_to_rust(left)?,
                op,
                self.expr_to_rust(right)?
            )),
        }
    }

    fn path_to_rust(&self, path: &[String]) -> Result<String, ScriptDiagnostic> {
        if path.is_empty() {
            return Err(ScriptDiagnostic::error("Empty path expression"));
        }

        if path.len() == 1 {
            return Ok(rust_ident(&path[0]));
        }

        if path[0] == "ctx" {
            if path == ["ctx", "dt"] {
                return Ok("ctx.dt".to_string());
            }
            if path.len() == 3 && path[1] == "state" {
                return Ok(format!("self.{}", state_field_ident(&path[2])));
            }
            if path.len() >= 3 && path[1] == "transform" {
                return transform_path_to_rust(&path[2..]);
            }
        }

        if self.locals.contains_key(&path[0]) {
            let mut out = rust_ident(&path[0]);
            for part in &path[1..] {
                out.push('.');
                out.push_str(part);
            }
            return Ok(out);
        }

        Err(ScriptDiagnostic::error(format!(
            "Unknown value '{}'",
            path.join(".")
        )))
    }

    fn call_to_rust(
        &mut self,
        callee: &[String],
        args: &[Expr],
    ) -> Result<String, ScriptDiagnostic> {
        match callee {
            [name] if name == "vec3" => {
                self.expect_arg_count(name, args, 3)?;
                Ok(format!(
                    "glam::Vec3::new({}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?
                ))
            }
            [name] if name == "quat_identity" => {
                self.expect_arg_count(name, args, 0)?;
                Ok("glam::Quat::IDENTITY".to_string())
            }
            [name] if name == "quat_from_rotation_y" => {
                self.expect_arg_count(name, args, 1)?;
                Ok(format!(
                    "glam::Quat::from_rotation_y({})",
                    self.expr_to_rust(&args[0])?
                ))
            }
            [name] if name == "sin" => {
                self.expect_arg_count(name, args, 1)?;
                Ok(format!("({}).sin()", self.expr_to_rust(&args[0])?))
            }
            [name] if name == "atan2" => {
                self.expect_arg_count(name, args, 2)?;
                Ok(format!(
                    "({}).atan2({})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?
                ))
            }
            [name] if name == "length" => {
                self.expect_arg_count(name, args, 1)?;
                Ok(format!("({}).length()", self.expr_to_rust(&args[0])?))
            }
            [name] if name == "length_squared" => {
                self.expect_arg_count(name, args, 1)?;
                Ok(format!(
                    "({}).length_squared()",
                    self.expr_to_rust(&args[0])?
                ))
            }
            [name] if name == "normalize" => {
                self.expect_arg_count(name, args, 1)?;
                let value = self.expr_to_rust(&args[0])?;
                Ok(format!(
                    "{{ let v = {}; if v.length_squared() > 0.000001 {{ v.normalize() }} else {{ glam::Vec3::ZERO }} }}",
                    value
                ))
            }
            [name] if name == "clamp" => {
                self.expect_arg_count(name, args, 3)?;
                Ok(format!(
                    "({}).clamp({}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?
                ))
            }
            [name] if name == "slerp" => {
                self.expect_arg_count(name, args, 3)?;
                Ok(format!(
                    "({}).slerp({}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "has_player" => {
                self.expect_arg_count(helper, args, 0)?;
                Ok("ctx.has_player()".to_string())
            }
            [ctx, helper] if ctx == "ctx" && helper == "player_position" => {
                self.expect_arg_count(helper, args, 0)?;
                Ok("ctx.player_position().unwrap_or(glam::Vec3::ZERO)".to_string())
            }
            [ctx, helper] if ctx == "ctx" && helper == "distance_to_player" => {
                self.expect_arg_count(helper, args, 0)?;
                Ok("ctx.distance_to_player()".to_string())
            }
            [ctx, helper] if ctx == "ctx" && helper == "track_player" => {
                self.expect_arg_count(helper, args, 4)?;
                Ok(format!(
                    "ctx.track_player({}, {}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?
                ))
            }
            [ctx, helper]
                if ctx == "ctx"
                    && matches!(
                        helper.as_str(),
                        "action_pressed" | "action_just_pressed" | "action_just_released"
                    ) =>
            {
                self.expect_arg_count(helper, args, 1)?;
                Ok(format!("ctx.{}({})", helper, self.expr_to_rust(&args[0])?))
            }
            [ctx, helper] if ctx == "ctx" && helper == "action_value" => {
                self.expect_arg_count(helper, args, 1)?;
                Ok(format!(
                    "ctx.action_value({})",
                    self.expr_to_rust(&args[0])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "set_procedural_generation" => {
                self.expect_arg_count(helper, args, 1)?;
                Ok(format!(
                    "ctx.set_procedural_generation({})",
                    self.expr_to_rust(&args[0])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "set_game_mode" => {
                self.expect_arg_count(helper, args, 1)?;
                Ok(format!(
                    "ctx.set_game_mode({})",
                    game_mode_arg_to_rust(&args[0])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "set_sandbox_clock" => {
                self.expect_arg_count(helper, args, 2)?;
                Ok(format!(
                    "ctx.set_sandbox_clock({}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "configure_sandbox" => {
                self.expect_arg_count(helper, args, 11)?;
                Ok(format!(
                    "ctx.configure_sandbox({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                    sandbox_profile_arg_to_rust(&args[0])?,
                    game_mode_arg_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?,
                    self.expr_to_rust(&args[7])?,
                    self.expr_to_rust(&args[8])?,
                    self.expr_to_rust(&args[9])?,
                    self.expr_to_rust(&args[10])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "configure_twilight_survival" => {
                self.expect_arg_count(helper, args, 7)?;
                Ok(format!(
                    "ctx.configure_twilight_survival({}, {}, {}, {}, {}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_primitive" => {
                self.expect_arg_count(helper, args, 7)?;
                Ok(format!(
                    "ctx.spawn_primitive({}, {}, {}, {}, {}, {}, {})",
                    primitive_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_ground" => {
                self.expect_arg_count(helper, args, 4)?;
                Ok(format!(
                    "ctx.spawn_ground({}, {}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_light" => {
                self.expect_arg_count(helper, args, 6)?;
                Ok(format!(
                    "ctx.spawn_light({}, {}, {}, {}, {}, {})",
                    light_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_light_with_cones" => {
                self.expect_arg_count(helper, args, 8)?;
                Ok(format!(
                    "ctx.spawn_light_with_cones({}, {}, {}, {}, {}, {}, {}, {})",
                    light_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?,
                    self.expr_to_rust(&args[7])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_scatter" => {
                self.expect_arg_count(helper, args, 10)?;
                Ok(format!(
                    "ctx.spawn_scatter({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                    primitive_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?,
                    self.expr_to_rust(&args[7])?,
                    self.expr_to_rust(&args[8])?,
                    self.expr_to_rust(&args[9])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_scatter_range" => {
                self.expect_arg_count(helper, args, 11)?;
                Ok(format!(
                    "ctx.spawn_scatter_range({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                    primitive_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?,
                    self.expr_to_rust(&args[7])?,
                    self.expr_to_rust(&args[8])?,
                    self.expr_to_rust(&args[9])?,
                    self.expr_to_rust(&args[10])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_grid" => {
                self.expect_arg_count(helper, args, 9)?;
                Ok(format!(
                    "ctx.spawn_grid({}, {}, {}, {}, {}, {}, {}, {}, {})",
                    primitive_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?,
                    self.expr_to_rust(&args[6])?,
                    self.expr_to_rust(&args[7])?,
                    self.expr_to_rust(&args[8])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_resource_cluster" => {
                self.expect_arg_count(helper, args, 5)?;
                Ok(format!(
                    "ctx.spawn_resource_cluster({}, {}, {}, {}, {})",
                    resource_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_tree_cluster" => {
                self.expect_arg_count(helper, args, 4)?;
                Ok(format!(
                    "ctx.spawn_tree_cluster({}, {}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_campfire" => {
                self.expect_arg_count(helper, args, 3)?;
                Ok(format!(
                    "ctx.spawn_campfire({}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "spawn_furry_npc" => {
                self.expect_arg_count(helper, args, 6)?;
                Ok(format!(
                    "ctx.spawn_furry_npc({}, {}, {}, {}, {}, {})",
                    furry_species_arg_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?,
                    self.expr_to_rust(&args[5])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "set_script" => {
                self.expect_arg_count(helper, args, 2)?;
                Ok(format!(
                    "ctx.set_script({}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?
                ))
            }
            [ctx, helper] if ctx == "ctx" && helper == "set_scripts" => {
                self.expect_arg_count(helper, args, 5)?;
                Ok(format!(
                    "ctx.set_scripts({}, {}, {}, {}, {})",
                    self.expr_to_rust(&args[0])?,
                    self.expr_to_rust(&args[1])?,
                    self.expr_to_rust(&args[2])?,
                    self.expr_to_rust(&args[3])?,
                    self.expr_to_rust(&args[4])?
                ))
            }
            _ => Err(ScriptDiagnostic::error(format!(
                "Unsupported call '{}'",
                callee.join(".")
            ))),
        }
    }

    fn expect_arg_count(
        &self,
        helper: &str,
        args: &[Expr],
        count: usize,
    ) -> Result<(), ScriptDiagnostic> {
        if args.len() == count {
            Ok(())
        } else {
            Err(ScriptDiagnostic::error(format!(
                "{} expects {} argument(s), got {}",
                helper,
                count,
                args.len()
            )))
        }
    }

    fn temp_name(&mut self) -> String {
        let index = self.temp_index;
        self.temp_index += 1;
        format!("__stfsc_value_{}", index)
    }

    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }
}

fn transform_path_to_rust(path: &[String]) -> Result<String, ScriptDiagnostic> {
    let Some(first) = path.first() else {
        return Err(ScriptDiagnostic::error("Expected ctx.transform field"));
    };

    let default = match first.as_str() {
        "position" => "glam::Vec3::ZERO",
        "scale" => "glam::Vec3::ONE",
        "rotation" => "glam::Quat::IDENTITY",
        _ => {
            return Err(ScriptDiagnostic::error(format!(
                "Unsupported transform field '{}'",
                first
            )));
        }
    };

    let field = path.join(".");
    if path.len() == 1 {
        Ok(format!(
            "ctx.transform().map(|t| t.{}).unwrap_or({})",
            field, default
        ))
    } else {
        Ok(format!(
            "ctx.transform().map(|t| t.{}).unwrap_or(0.0)",
            field
        ))
    }
}

fn hand_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value == "left" => Ok("XrHand::Left"),
        Expr::String(value) if value == "right" => Ok("XrHand::Right"),
        _ => Err(ScriptDiagnostic::error(
            "Expected hand string \"left\" or \"right\"",
        )),
    }
}

fn primitive_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value.eq_ignore_ascii_case("cube") => Ok("ScriptPrimitive::Cube"),
        Expr::String(value) if value.eq_ignore_ascii_case("sphere") => {
            Ok("ScriptPrimitive::Sphere")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("cylinder") => {
            Ok("ScriptPrimitive::Cylinder")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("plane") => Ok("ScriptPrimitive::Plane"),
        Expr::String(value) if value.eq_ignore_ascii_case("capsule") => {
            Ok("ScriptPrimitive::Capsule")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("cone") => Ok("ScriptPrimitive::Cone"),
        _ => Err(ScriptDiagnostic::error(
            "Expected primitive string \"cube\", \"sphere\", \"cylinder\", \"plane\", \"capsule\", or \"cone\"",
        )),
    }
}

fn light_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value.eq_ignore_ascii_case("point") => Ok("ScriptLightType::Point"),
        Expr::String(value) if value.eq_ignore_ascii_case("spot") => Ok("ScriptLightType::Spot"),
        Expr::String(value) if value.eq_ignore_ascii_case("directional") => {
            Ok("ScriptLightType::Directional")
        }
        _ => Err(ScriptDiagnostic::error(
            "Expected light string \"point\", \"spot\", or \"directional\"",
        )),
    }
}

fn game_mode_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value.eq_ignore_ascii_case("survival") => {
            Ok("ScriptGameMode::Survival")
        }
        Expr::String(value)
            if value.eq_ignore_ascii_case("god_mode") || value.eq_ignore_ascii_case("godmode") =>
        {
            Ok("ScriptGameMode::GodMode")
        }
        _ => Err(ScriptDiagnostic::error(
            "Expected game mode string \"survival\" or \"god_mode\"",
        )),
    }
}

fn sandbox_profile_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value)
            if value.eq_ignore_ascii_case("forest_survival")
                || value.eq_ignore_ascii_case("forest") =>
        {
            Ok("ScriptSandboxProfile::ForestSurvival")
        }
        Expr::String(value)
            if value.eq_ignore_ascii_case("urban_streaming")
                || value.eq_ignore_ascii_case("urban") =>
        {
            Ok("ScriptSandboxProfile::UrbanStreaming")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("hybrid") => {
            Ok("ScriptSandboxProfile::Hybrid")
        }
        _ => Err(ScriptDiagnostic::error(
            "Expected sandbox profile string \"forest_survival\", \"urban_streaming\", or \"hybrid\"",
        )),
    }
}

fn resource_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value.eq_ignore_ascii_case("wood") => Ok("ScriptResourceKind::Wood"),
        Expr::String(value) if value.eq_ignore_ascii_case("stone") => {
            Ok("ScriptResourceKind::Stone")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("ore") => Ok("ScriptResourceKind::Ore"),
        Expr::String(value) if value.eq_ignore_ascii_case("crystal") => {
            Ok("ScriptResourceKind::Crystal")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("fiber") => {
            Ok("ScriptResourceKind::Fiber")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("food") => Ok("ScriptResourceKind::Food"),
        _ => Err(ScriptDiagnostic::error(
            "Expected resource string \"wood\", \"stone\", \"ore\", \"crystal\", \"fiber\", or \"food\"",
        )),
    }
}

fn furry_species_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value.eq_ignore_ascii_case("fox") => Ok("ScriptFurrySpecies::Fox"),
        Expr::String(value) if value.eq_ignore_ascii_case("wolf") => Ok("ScriptFurrySpecies::Wolf"),
        Expr::String(value) if value.eq_ignore_ascii_case("cat") => Ok("ScriptFurrySpecies::Cat"),
        Expr::String(value) if value.eq_ignore_ascii_case("rabbit") => {
            Ok("ScriptFurrySpecies::Rabbit")
        }
        Expr::String(value) if value.eq_ignore_ascii_case("bear") => Ok("ScriptFurrySpecies::Bear"),
        _ => Err(ScriptDiagnostic::error(
            "Expected furry species string \"fox\", \"wolf\", \"cat\", \"rabbit\", or \"bear\"",
        )),
    }
}

fn pose_arg_to_rust(expr: &Expr) -> Result<&'static str, ScriptDiagnostic> {
    match expr {
        Expr::String(value) if value == "head" => Ok("XrPoseSpace::Head"),
        Expr::String(value) if value == "left_grip" => Ok("XrPoseSpace::Grip(XrHand::Left)"),
        Expr::String(value) if value == "right_grip" => Ok("XrPoseSpace::Grip(XrHand::Right)"),
        Expr::String(value) if value == "left_aim" => Ok("XrPoseSpace::Aim(XrHand::Left)"),
        Expr::String(value) if value == "right_aim" => Ok("XrPoseSpace::Aim(XrHand::Right)"),
        _ => Err(ScriptDiagnostic::error(
            "Expected XR pose string like \"head\" or \"right_grip\"",
        )),
    }
}

fn number_to_f32(value: &str) -> String {
    if value.contains('.') {
        format!("{}f32", value)
    } else {
        format!("{}.0f32", value)
    }
}

pub fn rust_type_name(raw: &str) -> String {
    let mut out = rust_ident(raw);
    if out.is_empty() {
        out.push_str("GeneratedScript");
    }
    if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

pub fn runtime_cache_name(source_hash: u64) -> String {
    format!("__stfsc_script_{:016x}", source_hash)
}

fn state_field_ident(name: &str) -> String {
    format!("state_{}", rust_ident(name))
}

fn rust_ident(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if is_rust_keyword(&out) {
        out.push('_');
    }
    out
}

fn is_rust_keyword(value: &str) -> bool {
    matches!(
        value,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
    )
}

fn matches_xyz(value: &str) -> bool {
    matches!(value, "x" | "y" | "z")
}

fn matches_xyzw(value: &str) -> bool {
    matches!(value, "x" | "y" | "z" | "w")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str) -> String {
        render_native_script_module(&[NativeScriptSource {
            runtime_name: "__test".to_string(),
            struct_name: "GeneratedTest".to_string(),
            source: source.to_string(),
        }])
        .expect("script compiles")
    }

    #[test]
    fn compiles_bounce_script_to_transform_write() {
        let rust = render(
            r#"
script Bounce {
    on_start(ctx) {
        ctx.state.time = 0.0;
        ctx.state.base_y = ctx.transform.position.y;
    }

    on_update(ctx) {
        ctx.state.time = ctx.state.time + ctx.dt;
        let p = ctx.transform.position;
        let y = ctx.state.base_y + sin(ctx.state.time * 2.0) * 0.5;
        ctx.transform.position = vec3(p.x, y, p.z);
    }
}
"#,
        );

        assert!(rust.contains("pub struct GeneratedTest"));
        assert!(rust.contains("state_time"));
        assert!(rust.contains("transform.position"));
        assert!(rust.contains("glam::Vec3::new"));
    }

    #[test]
    fn compiles_haptic_and_xr_pose_helpers() {
        let rust = render(
            r#"
script FollowHand {
    on_late_update(ctx) {
        ctx.track_transform_to_xr_pose("right_grip", vec3(0.0, 0.0, 0.0), quat_identity());
    }

    on_update(ctx) {
        if ctx.action_just_pressed("fire") {
            ctx.request_haptic("right", 0.7, 0.08);
        }
    }
}
"#,
        );

        assert!(rust.contains("XrPoseSpace::Grip(XrHand::Right)"));
        assert!(rust.contains("ctx.request_haptic(XrHand::Right"));
    }

    #[test]
    fn compiles_projectile_helpers_for_ranged_weapons() {
        let rust = render(
            r#"
script CachedRangedWeapon {
    on_update(ctx) {
        if ctx.action_just_pressed("fire") {
            let forward = ctx.transform.rotation * vec3(0.0, 0.0, -1.0);
            let origin = ctx.transform.position + forward * 0.5;
            ctx.fire_projectile(origin, forward, 48.0, 3.0, 0.05, 10.0, 0.0);
        }

        if ctx.action_just_released("fire") {
            ctx.fire_projectile_from_self(30.0, 6.0, 0.035, 18.0, 1.0);
        }
    }
}
"#,
        );

        assert!(rust.contains("ctx.fire_projectile("));
        assert!(rust.contains("ctx.fire_projectile_from_self("));
    }

    #[test]
    fn compiles_enemy_tracking_helpers_to_native_cache_calls() {
        let rust = render(
            r#"
script CachedEnemyTracker {
    on_update(ctx) {
        let radius = 18.0;
        let speed = 3.5;
        let stop = 1.25;
        let turn = 9.0;

        if ctx.has_player() && ctx.distance_to_player() < radius {
            let player = ctx.player_position();
            ctx.track_player(radius, speed, stop, turn);
            ctx.log("tracking player");
        }
    }
}
"#,
        );

        assert!(rust.contains("ctx.has_player()"));
        assert!(rust.contains("ctx.distance_to_player()"));
        assert!(rust.contains("ctx.player_position().unwrap_or(glam::Vec3::ZERO)"));
        assert!(rust.contains("ctx.track_player(radius, speed, stop, turn)"));
    }

    #[test]
    fn compiles_scene_procedural_helpers() {
        let rust = render(
            r#"
script SceneGenerator {
    on_start(ctx) {
        ctx.set_procedural_generation(false);
        ctx.configure_twilight_survival(8842.0, 48.0, 8.0, 0.22, 0.18, 0.04, 0.08);
        ctx.set_game_mode("survival");
        ctx.set_sandbox_clock(0.68, 0.0);
        ctx.spawn_ground(vec3(0.0, -0.5, 0.0), vec3(80.0, 1.0, 80.0), vec3(0.2, 0.5, 0.25), true);
        let id = ctx.spawn_primitive("cube", vec3(0.0, 1.0, -4.0), quat_identity(), vec3(2.0, 2.0, 2.0), vec3(0.9, 0.4, 0.2), true, true);
        ctx.set_script(id, "TestBounce");
        ctx.spawn_light("point", vec3(0.0, 4.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(1.0, 0.9, 0.75), 3.0, 20.0);
        ctx.spawn_light_with_cones("spot", vec3(4.0, 6.0, 2.0), vec3(-1.0, -1.0, 0.0), vec3(0.6, 0.7, 1.0), 8.0, 24.0, 0.25, 0.85);
        ctx.spawn_scatter("sphere", 12.0, vec3(0.0, 0.5, 0.0), vec3(20.0, 0.0, 20.0), vec3(0.4, 0.4, 0.4), vec3(1.1, 1.1, 1.1), vec3(0.4, 0.7, 0.5), 33.0, true, true);
        ctx.spawn_scatter_range("cone", 9.0, vec3(0.0, 0.5, 0.0), vec3(12.0, 0.0, 12.0), vec3(0.5, 0.5, 0.5), vec3(1.5, 2.0, 1.5), vec3(0.25, 0.5, 0.35), vec3(0.8, 0.9, 0.45), 77.0, false, true);
        ctx.spawn_grid("cube", 4.0, 3.0, vec3(0.0, 0.25, 8.0), vec3(2.0, 0.0, 2.0), vec3(1.0, 0.5, 1.0), vec3(0.2, 0.3, 0.8), true, true);
        ctx.set_scripts(id, "TestBounce", "CollisionLogger", "", "");
        ctx.spawn_resource_cluster("crystal", 18.0, vec3(8.0, 0.4, -8.0), vec3(18.0, 0.0, 18.0), 101.0);
        ctx.spawn_tree_cluster(12.0, vec3(-12.0, 0.0, 10.0), vec3(24.0, 0.0, 24.0), 202.0);
        ctx.spawn_campfire(vec3(0.0, 0.0, 0.0), 1.0, true);
        let friend = ctx.spawn_furry_npc("fox", vec3(2.0, 0.0, 2.0), vec3(0.9, 0.5, 0.22), 1.0, "CrowdAgent", "");
        ctx.set_script(friend, "CrowdAgent");
    }
}
"#,
        );

        assert!(rust.contains("ScriptPrimitive::Cube"));
        assert!(rust.contains("ScriptLightType::Point"));
        assert!(rust.contains("ctx.spawn_light_with_cones(ScriptLightType::Spot"));
        assert!(rust.contains("ctx.spawn_scatter(ScriptPrimitive::Sphere"));
        assert!(rust.contains("ctx.spawn_scatter_range(ScriptPrimitive::Cone"));
        assert!(rust.contains("ctx.spawn_grid(ScriptPrimitive::Cube"));
        assert!(rust.contains("ctx.set_script(id, \"TestBounce\")"));
        assert!(
            rust.contains("ctx.set_scripts(id, \"TestBounce\", \"CollisionLogger\", \"\", \"\")")
        );
        assert!(rust.contains("ctx.configure_twilight_survival("));
        assert!(rust.contains("ctx.set_game_mode(ScriptGameMode::Survival)"));
        assert!(rust.contains("ctx.spawn_resource_cluster(ScriptResourceKind::Crystal"));
        assert!(rust.contains("ctx.spawn_tree_cluster("));
        assert!(rust.contains("ctx.spawn_campfire("));
        assert!(rust.contains("ctx.spawn_furry_npc(ScriptFurrySpecies::Fox"));
    }

    #[test]
    fn rejects_unknown_helpers() {
        let err = render_native_script_module(&[NativeScriptSource {
            runtime_name: "__bad".to_string(),
            struct_name: "Bad".to_string(),
            source: r#"
script Bad {
    on_update(ctx) {
        ctx.teleport_to_player();
    }
}
"#
            .to_string(),
        }])
        .expect_err("unknown helper should fail");

        assert!(err
            .iter()
            .any(|diag| diag.message.contains("Unsupported call")));
    }
}
