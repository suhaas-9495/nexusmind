"""NexusMind – Calculator Tool"""

import ast
import operator

# Safe operator whitelist
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression type: {type(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        tree   = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error for `{expression}`: {e}"
