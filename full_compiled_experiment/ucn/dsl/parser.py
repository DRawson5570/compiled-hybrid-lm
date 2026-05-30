from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

from .ast import (
    Activate,
    ActivateType,
    AllocDecl,
    DBSpec,
    GatherContext,
    MatrixRef,
    Mix,
    Program,
    Project,
    QueryMemory,
    Residual,
    Rotate,
    ScalarExpr,
    Statement,
    SubspaceRef,
    Transform,
    TypeSpec,
)


class TokenType(Enum):
    ALLOC = "alloc"
    ASSIGN = "="
    SEMICOLON = ";"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    COLON = ":"
    DOT = "."
    IDENT = "ident"
    INTEGER = "int"
    FLOAT = "float"
    EOF = "eof"


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


class ParserError(Exception):
    pass


def tokenize(source: str) -> list[Token]:
    spec = [
        (r"alloc", TokenType.ALLOC),
        (r"=", TokenType.ASSIGN),
        (r";", TokenType.SEMICOLON),
        (r"\(", TokenType.LPAREN),
        (r"\)", TokenType.RPAREN),
        (r"\[", TokenType.LBRACKET),
        (r"\]", TokenType.RBRACKET),
        (r",", TokenType.COMMA),
        (r":", TokenType.COLON),
        (r"\.", TokenType.DOT),
        (r"\d+\.\d+", TokenType.FLOAT),
        (r"\d+", TokenType.INTEGER),
        (r"[a-zA-Z_][a-zA-Z0-9_]*", TokenType.IDENT),
    ]

    tokens = []
    pos = 0
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue

        matched = False
        for pattern, ttype in spec:
            m = re.match(pattern, source[pos:])
            if m:
                val = m.group(0)
                tokens.append(Token(ttype, val, pos))
                pos += len(val)
                matched = True
                break

        if not matched:
            raise ParserError(
                f"Unexpected character '{source[pos]}' at position {pos}"
            )

    tokens.append(Token(TokenType.EOF, "", pos))
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, ttype: TokenType) -> Token:
        token = self.peek()
        if token.type != ttype:
            raise ParserError(
                f"Expected {ttype.value} but got {token.type.value} "
                f"at position {token.pos}"
            )
        return self.advance()

    def parse_program(self) -> Program:
        program = Program()
        while self.peek().type == TokenType.ALLOC:
            program.declarations.append(self._parse_declaration())
            if self.peek().type == TokenType.SEMICOLON:
                self.advance()

        while self.peek().type != TokenType.EOF:
            program.statements.append(self._parse_statement())
            if self.peek().type == TokenType.SEMICOLON:
                self.advance()

        return program

    def _parse_declaration(self) -> AllocDecl:
        self.expect(TokenType.ALLOC)
        self.expect(TokenType.LPAREN)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        type_spec = self._parse_type_spec()
        self.expect(TokenType.RPAREN)
        return AllocDecl(name, type_spec[0], type_spec[1])

    def _parse_type_spec(self) -> tuple[TypeSpec, int]:
        base = self.expect(TokenType.IDENT).value
        tspec = {
            "Vector": TypeSpec.VECTOR,
            "Subspace": TypeSpec.SUBSPACE,
            "Scalar": TypeSpec.SCALAR,
            "Matrix": TypeSpec.MATRIX,
        }.get(base)
        if tspec is None:
            raise ParserError(f"Unknown type: {base}")

        dim = 0
        if self.peek().type == TokenType.LBRACKET:
            self.advance()
            dim = int(self.expect(TokenType.INTEGER).value)
            self.expect(TokenType.RBRACKET)

        return tspec, dim

    def _parse_statement(self) -> Statement:
        target = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        expr = self._parse_expr()
        return Statement(target, expr)

    def _parse_expr(self) -> Any:
        return self._expect_op()

    def _expect_op(self):
        name = self.expect(TokenType.IDENT).value

        handlers = {
            "mix": self._parse_mix,
            "project": self._parse_project,
            "transform": self._parse_transform,
            "activate": self._parse_activate,
            "query_memory": self._parse_query_memory,
            "residual": self._parse_residual,
            "rotate": self._parse_rotate,
            "gather_context": self._parse_gather_context,
        }

        handler = handlers.get(name)
        if handler is None:
            raise ParserError(f"Unknown operation: {name}")
        return handler(name)

    def _parse_mix(self, op_name: str) -> Mix:
        self.expect(TokenType.LPAREN)
        names = self._parse_id_list()
        self.expect(TokenType.COMMA)
        weights = self._parse_scalar_list()
        self.expect(TokenType.RPAREN)
        return Mix(names, weights)

    def _parse_project(self, op_name: str) -> Project:
        self.expect(TokenType.LPAREN)
        ident = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        subspace = self._parse_subspace_ref()
        self.expect(TokenType.RPAREN)
        return Project(ident, subspace)

    def _parse_transform(self, op_name: str) -> Transform:
        self.expect(TokenType.LPAREN)
        ident = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        matrix = self._parse_matrix_ref()
        self.expect(TokenType.RPAREN)
        return Transform(ident, matrix)

    def _parse_activate(self, op_name: str) -> Activate:
        self.expect(TokenType.LPAREN)
        ident = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        act_str = self.expect(TokenType.IDENT).value
        act = {
            "gelu": ActivateType.GELU,
            "relu": ActivateType.RELU,
            "silu": ActivateType.SILU,
            "identity": ActivateType.IDENTITY,
        }.get(act_str)
        if act is None:
            raise ParserError(f"Unknown activation: {act_str}")
        self.expect(TokenType.RPAREN)
        return Activate(ident, act)

    def _parse_query_memory(self, op_name: str) -> QueryMemory:
        self.expect(TokenType.LPAREN)
        ident = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        db = self._parse_db_spec()
        self.expect(TokenType.COMMA)
        top_k = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.RPAREN)
        return QueryMemory(ident, db, top_k)

    def _parse_residual(self, op_name: str) -> Residual:
        self.expect(TokenType.LPAREN)
        names = self._parse_id_list()
        self.expect(TokenType.RPAREN)
        return Residual(names)

    def _parse_rotate(self, op_name: str) -> Rotate:
        self.expect(TokenType.LPAREN)
        ident = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        theta = self._parse_scalar_val()
        self.expect(TokenType.COMMA)
        subspace = self._parse_subspace_ref()
        self.expect(TokenType.RPAREN)
        return Rotate(ident, theta, subspace)

    def _parse_gather_context(self, op_name: str) -> GatherContext:
        self.expect(TokenType.LPAREN)
        query = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        source = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COMMA)
        top_k = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.RPAREN)
        return GatherContext(query=query, source=source, top_k=top_k)

    def _parse_id_list(self) -> list[str]:
        self.expect(TokenType.LBRACKET)
        names = [self.expect(TokenType.IDENT).value]
        while self.peek().type == TokenType.COMMA:
            self.advance()
            names.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RBRACKET)
        return names

    def _parse_scalar_list(self) -> list[float]:
        self.expect(TokenType.LBRACKET)
        vals = [self._parse_scalar_val()]
        while self.peek().type == TokenType.COMMA:
            self.advance()
            vals.append(self._parse_scalar_val())
        self.expect(TokenType.RBRACKET)
        return vals

    def _parse_scalar_val(self) -> float:
        token = self.peek()
        if token.type == TokenType.FLOAT:
            self.advance()
            return float(token.value)
        elif token.type == TokenType.INTEGER:
            self.advance()
            return float(token.value)
        else:
            raise ParserError(f"Expected scalar, got {token.type.value}")

    def _parse_subspace_ref(self) -> SubspaceRef:
        self.expect(TokenType.LBRACKET)
        start = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.COLON)
        end = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.RBRACKET)
        return SubspaceRef(start, end)

    def _parse_matrix_ref(self) -> MatrixRef:
        prefix = self.expect(TokenType.IDENT).value
        self.expect(TokenType.DOT)
        name = self.expect(TokenType.IDENT).value
        return MatrixRef(prefix, name)

    def _parse_db_spec(self) -> DBSpec:
        self.expect(TokenType.IDENT)
        self.expect(TokenType.DOT)
        partition = self.expect(TokenType.IDENT).value
        return DBSpec(partition)


def parse_program(source: str) -> Program:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
