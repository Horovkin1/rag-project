"""Безопасный калькулятор — пример простого инструмента."""
import ast
import operator

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(round(result, 10))
    except Exception as e:
        return f"Ошибка: {e}"


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Оператор не поддерживается: {node.op}")
        return op(_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Унарный оператор не поддерживается")
        return op(_eval(node.operand))
    raise ValueError(f"Неподдерживаемый тип: {type(node)}")
