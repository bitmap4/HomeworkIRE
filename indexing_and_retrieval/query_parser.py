import re
from typing import List, Set, Union
from enum import Enum

class TokenType(Enum):
    TERM = "TERM"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"

class Token:
    def __init__(self, type: TokenType, value: str):
        self.type = type
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Lexer:
    def __init__(self, query: str):
        self.query = query
        self.pos = 0
        self.current_char = self.query[0] if query else None
    
    def advance(self):
        self.pos += 1
        self.current_char = self.query[self.pos] if self.pos < len(self.query) else None
    
    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def read_quoted_string(self) -> str:
        result = ''
        self.advance()
        
        while self.current_char and self.current_char != '"':
            result += self.current_char
            self.advance()
        
        if self.current_char == '"':
            self.advance()
        
        return result
    
    def read_word(self) -> str:
        result = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result
    
    def tokenize(self) -> List[Token]:
        tokens = []
        
        while self.current_char:
            self.skip_whitespace()
            
            if not self.current_char:
                break
            
            if self.current_char == '"':
                term = self.read_quoted_string()
                tokens.append(Token(TokenType.TERM, term))
            elif self.current_char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                self.advance()
            elif self.current_char.isalpha():
                word = self.read_word()
                if word == 'AND':
                    tokens.append(Token(TokenType.AND, 'AND'))
                elif word == 'OR':
                    tokens.append(Token(TokenType.OR, 'OR'))
                elif word == 'NOT':
                    tokens.append(Token(TokenType.NOT, 'NOT'))
                else:
                    tokens.append(Token(TokenType.TERM, word))
            else:
                self.advance()
        
        tokens.append(Token(TokenType.EOF, ''))
        return tokens

class ASTNode:
    pass

class TermNode(ASTNode):
    def __init__(self, term: str):
        self.term = term
    
    def __repr__(self):
        return f"Term({self.term})"

class PhraseNode(ASTNode):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def __repr__(self):
        return f"Phrase({self.phrase})"

class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"{self.op}({self.left}, {self.right})"

class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand
    
    def __repr__(self):
        return f"{self.op}({self.operand})"

class QueryParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self):
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def parse(self) -> ASTNode:
        return self.parse_or()
    
    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        
        while self.current_token and self.current_token.type == TokenType.OR:
            self.advance()
            right = self.parse_and()
            left = BinaryOpNode('OR', left, right)
        
        return left
    
    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        
        while self.current_token and self.current_token.type == TokenType.AND:
            self.advance()
            right = self.parse_not()
            left = BinaryOpNode('AND', left, right)
        
        return left
    
    def parse_not(self) -> ASTNode:
        if self.current_token and self.current_token.type == TokenType.NOT:
            self.advance()
            operand = self.parse_not()
            return UnaryOpNode('NOT', operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            node = self.parse_or()
            if self.current_token.type == TokenType.RPAREN:
                self.advance()
            return node
        
        elif self.current_token.type == TokenType.TERM:
            term = self.current_token.value
            self.advance()
            
            if ' ' in term or len(term.split()) > 1:
                return PhraseNode(term)
            else:
                return TermNode(term)
        
        else:
            raise ValueError(f"Unexpected token: {self.current_token}")

def parse_query(query: str) -> ASTNode:
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = QueryParser(tokens)
    return parser.parse()
