"""
Metadata filtering for search results.

Supports structured metadata fields and filter expressions with AND, OR, NOT operators.
"""
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import duckdb
from loguru import logger


class FieldType(Enum):
    """Supported metadata field types."""
    TEXT = "text"
    DATE = "date"
    NUMBER = "number"
    ARRAY = "array"
    BOOLEAN = "boolean"


@dataclass
class MetadataSchema:
    """Schema definition for metadata fields."""
    fields: Dict[str, FieldType] = field(default_factory=dict)
    
    def validate(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema."""
        for key, value in metadata.items():
            if key in self.fields:
                expected_type = self.fields[key]
                if not self._check_type(value, expected_type):
                    return False
        return True
    
    def _check_type(self, value: Any, expected: FieldType) -> bool:
        if expected == FieldType.TEXT:
            return isinstance(value, str)
        elif expected == FieldType.NUMBER:
            return isinstance(value, (int, float))
        elif expected == FieldType.BOOLEAN:
            return isinstance(value, bool)
        elif expected == FieldType.ARRAY:
            return isinstance(value, list)
        elif expected == FieldType.DATE:
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return True
                except ValueError:
                    return False
            return isinstance(value, datetime)
        return True


class FilterOperator(Enum):
    """Filter comparison operators."""
    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    CONTAINS = "~"
    IN = "in"
    ALL = "all"
    ANY = "any"


@dataclass
class FilterCondition:
    """Single filter condition."""
    field: str
    operator: FilterOperator
    value: Any


@dataclass
class FilterAST:
    """Abstract syntax tree for filter expressions."""
    pass


@dataclass
class ConditionNode(FilterAST):
    """Leaf node: single condition."""
    condition: FilterCondition


@dataclass
class AndNode(FilterAST):
    """AND combination of filters."""
    children: List[FilterAST]


@dataclass
class OrNode(FilterAST):
    """OR combination of filters."""
    children: List[FilterAST]


@dataclass
class NotNode(FilterAST):
    """NOT negation of filter."""
    child: FilterAST


class MetadataStore:
    """Store and retrieve document metadata using DuckDB."""
    
    def __init__(self, db_path: str = "index.duckdb"):
        self.db_path = db_path
    
    def set(self, doc_id: int, metadata: Dict[str, Any]) -> None:
        """Store metadata for a document."""
        with duckdb.connect(self.db_path) as con:
            metadata_json = json.dumps(metadata)
            con.execute(
                "UPDATE docs SET metadata = ? WHERE doc_id = ?",
                [metadata_json, doc_id]
            )
            logger.debug(f"Set metadata for doc_id={doc_id}")
    
    def get(self, doc_id: int) -> Dict[str, Any]:
        """Retrieve metadata for a document."""
        with duckdb.connect(self.db_path) as con:
            result = con.execute(
                "SELECT metadata FROM docs WHERE doc_id = ?",
                [doc_id]
            ).fetchone()
            if result and result[0]:
                return json.loads(result[0])
            return {}
    
    def set_batch(self, metadata_map: Dict[int, Dict[str, Any]]) -> None:
        """Store metadata for multiple documents."""
        with duckdb.connect(self.db_path) as con:
            for doc_id, metadata in metadata_map.items():
                metadata_json = json.dumps(metadata)
                con.execute(
                    "UPDATE docs SET metadata = ? WHERE doc_id = ?",
                    [metadata_json, doc_id]
                )
            logger.info(f"Set metadata for {len(metadata_map)} documents")
    
    def query(self, sql_where: str) -> List[int]:
        """Execute SQL query, return matching doc_ids."""
        with duckdb.connect(self.db_path) as con:
            query = f"SELECT doc_id FROM docs WHERE {sql_where}"
            result = con.execute(query).fetchall()
            return [row[0] for row in result]


class FilterParser:
    """Parse filter expressions into AST."""
    
    # Token patterns
    FIELD_PATTERN = r'([a-zA-Z_][a-zA-Z0-9_]*)'
    OPERATOR_PATTERN = r'(>=|<=|!=|>|<|=|~)'
    VALUE_PATTERN = r'([^\s\)]+|\[[^\]]*\]|"[^"]*")'
    
    def parse(self, filter_expr: str) -> FilterAST:
        """
        Parse filter expression into AST.
        
        Syntax examples:
        - field:value (exact match)
        - field:>value (comparison)
        - field:[v1,v2] (array contains any)
        - NOT field:value
        - field:value AND field2:value2
        - field:value OR field2:value2
        - (field:value AND field2:value2) OR field3:value3
        """
        filter_expr = filter_expr.strip()
        if not filter_expr:
            raise ValueError("Empty filter expression")
        
        return self._parse_or(filter_expr)
    
    def _parse_or(self, expr: str) -> FilterAST:
        """Parse OR expressions."""
        parts = self._split_by_operator(expr, ' OR ')
        if len(parts) == 1:
            return self._parse_and(parts[0])
        return OrNode([self._parse_and(p.strip()) for p in parts])
    
    def _parse_and(self, expr: str) -> FilterAST:
        """Parse AND expressions."""
        parts = self._split_by_operator(expr, ' AND ')
        if len(parts) == 1:
            return self._parse_not(parts[0])
        return AndNode([self._parse_not(p.strip()) for p in parts])
    
    def _parse_not(self, expr: str) -> FilterAST:
        """Parse NOT expressions."""
        expr = expr.strip()
        if expr.startswith('NOT '):
            return NotNode(self._parse_atom(expr[4:].strip()))
        return self._parse_atom(expr)
    
    def _parse_atom(self, expr: str) -> FilterAST:
        """Parse atomic condition or parenthesized expression."""
        expr = expr.strip()
        
        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_or(expr[1:-1])
        
        # Parse field:operator:value or field:value
        match = re.match(r'(\w+):(>=|<=|!=|>|<|=|~)?(.+)', expr)
        if not match:
            raise ValueError(f"Invalid filter syntax: {expr}")
        
        field = match.group(1)
        operator_str = match.group(2) or '='
        value_str = match.group(3).strip()
        
        # Parse operator
        operator = FilterOperator(operator_str)
        
        # Parse value
        value = self._parse_value(value_str)
        
        # Handle array operators
        if isinstance(value, list):
            operator = FilterOperator.ANY
        
        condition = FilterCondition(field=field, operator=operator, value=value)
        return ConditionNode(condition)
    
    def _parse_value(self, value_str: str) -> Union[str, int, float, bool, List[str]]:
        """Parse value string into appropriate type."""
        value_str = value_str.strip()
        
        # Array: [v1,v2,v3]
        if value_str.startswith('[') and value_str.endswith(']'):
            inner = value_str[1:-1]
            return [v.strip().strip('"\'') for v in inner.split(',')]
        
        # Quoted string
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # String
        return value_str
    
    def _split_by_operator(self, expr: str, operator: str) -> List[str]:
        """Split expression by operator, respecting parentheses."""
        parts = []
        current = []
        depth = 0
        i = 0
        
        while i < len(expr):
            if expr[i] == '(':
                depth += 1
                current.append(expr[i])
            elif expr[i] == ')':
                depth -= 1
                current.append(expr[i])
            elif depth == 0 and expr[i:i+len(operator)] == operator:
                parts.append(''.join(current))
                current = []
                i += len(operator) - 1
            else:
                current.append(expr[i])
            i += 1
        
        if current:
            parts.append(''.join(current))
        
        return parts


class MetadataFilter:
    """Apply metadata filters to search results."""
    
    def __init__(self, schema: Optional[MetadataSchema] = None):
        self.schema = schema
        self.parser = FilterParser()
    
    def parse(self, filter_expr: str) -> FilterAST:
        """Parse filter expression into AST."""
        return self.parser.parse(filter_expr)
    
    def apply(
        self,
        filter_ast: FilterAST,
        doc_ids: List[int],
        metadata_store: MetadataStore
    ) -> List[int]:
        """Apply filter to document IDs, return matching IDs."""
        matching = []
        for doc_id in doc_ids:
            metadata = metadata_store.get(doc_id)
            if self._evaluate(filter_ast, metadata):
                matching.append(doc_id)
        return matching
    
    def to_sql(self, filter_ast: FilterAST) -> str:
        """Convert filter to SQL WHERE clause for DuckDB."""
        return self._ast_to_sql(filter_ast)
    
    def _evaluate(self, ast: FilterAST, metadata: Dict[str, Any]) -> bool:
        """Evaluate filter AST against metadata."""
        if isinstance(ast, ConditionNode):
            return self._evaluate_condition(ast.condition, metadata)
        elif isinstance(ast, AndNode):
            return all(self._evaluate(child, metadata) for child in ast.children)
        elif isinstance(ast, OrNode):
            return any(self._evaluate(child, metadata) for child in ast.children)
        elif isinstance(ast, NotNode):
            return not self._evaluate(ast.child, metadata)
        return False
    
    def _evaluate_condition(self, cond: FilterCondition, metadata: Dict[str, Any]) -> bool:
        """Evaluate single condition against metadata."""
        value = metadata.get(cond.field)
        if value is None:
            return False
        
        target = cond.value
        op = cond.operator
        
        if op == FilterOperator.EQ:
            return value == target
        elif op == FilterOperator.NE:
            return value != target
        elif op == FilterOperator.GT:
            return value > target
        elif op == FilterOperator.GE:
            return value >= target
        elif op == FilterOperator.LT:
            return value < target
        elif op == FilterOperator.LE:
            return value <= target
        elif op == FilterOperator.CONTAINS:
            return str(target).lower() in str(value).lower()
        elif op == FilterOperator.ANY:
            if isinstance(value, list):
                return any(v in value for v in target)
            return value in target
        elif op == FilterOperator.ALL:
            if isinstance(value, list):
                return all(v in value for v in target)
            return False
        
        return False
    
    def _ast_to_sql(self, ast: FilterAST) -> str:
        """Convert AST to SQL WHERE clause."""
        if isinstance(ast, ConditionNode):
            return self._condition_to_sql(ast.condition)
        elif isinstance(ast, AndNode):
            parts = [self._ast_to_sql(child) for child in ast.children]
            return f"({' AND '.join(parts)})"
        elif isinstance(ast, OrNode):
            parts = [self._ast_to_sql(child) for child in ast.children]
            return f"({' OR '.join(parts)})"
        elif isinstance(ast, NotNode):
            return f"NOT ({self._ast_to_sql(ast.child)})"
        return "1=1"
    
    def _condition_to_sql(self, cond: FilterCondition) -> str:
        """Convert condition to SQL."""
        field = cond.field
        value = cond.value
        op = cond.operator
        
        # Use JSON extraction for metadata fields
        json_path = f"json_extract_string(metadata, '$.{field}')"
        
        if op == FilterOperator.EQ:
            return f"{json_path} = '{value}'"
        elif op == FilterOperator.NE:
            return f"{json_path} != '{value}'"
        elif op == FilterOperator.GT:
            return f"CAST({json_path} AS DOUBLE) > {value}"
        elif op == FilterOperator.GE:
            return f"CAST({json_path} AS DOUBLE) >= {value}"
        elif op == FilterOperator.LT:
            return f"CAST({json_path} AS DOUBLE) < {value}"
        elif op == FilterOperator.LE:
            return f"CAST({json_path} AS DOUBLE) <= {value}"
        elif op == FilterOperator.CONTAINS:
            return f"{json_path} LIKE '%{value}%'"
        elif op in (FilterOperator.ANY, FilterOperator.IN):
            if isinstance(value, list):
                values = "', '".join(str(v) for v in value)
                return f"{json_path} IN ('{values}')"
            return f"{json_path} = '{value}'"
        
        return "1=1"
