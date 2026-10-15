"""第一课：制作、检查训练样本。只使用 Python 标准库，不调用模型。"""

import json
from pathlib import Path


RULES = (
    "你是工具调用助手。计算题调用 calculate，参数 expression 为字符串。"
    "天气查询调用 get_weather，参数 city 为字符串；缺少城市时追问。"
    "概念问题直接回答。输出 JSON，action 只能是 tool、clarify、answer。"
    "tool 包含 name 和 arguments；clarify 和 answer 包含 content。"
)

# 人工确定的正确动作叫标签（label），不是模型预测结果。
CASES = [
    {
        "id": "calculate",
        "user": "帮我算一下 (128 + 64) / 3。",
        "expected": {"action": "tool", "name": "calculate",
                     "arguments": {"expression": "(128 + 64) / 3"}},
        "bad": {"action": "tool", "name": "get_weather",
                "arguments": {"city": "上海"}},
        "reason": "这是计算题，却选择了天气工具。",
    },
    {
        "id": "clarify",
        "user": "今天会下雨吗？",
        "expected": {"action": "clarify", "content": "你想查询哪个城市的天气？"},
        "bad": {"action": "tool", "name": "get_weather",
                "arguments": {"city": "北京"}},
        "reason": "上下文没有城市信息，却自行填入北京。",
    },
    {
        "id": "answer",
        "user": "什么是模型微调？",
        "expected": {"action": "answer",
                     "content": "模型微调是在已有模型基础上，用特定任务的数据继续训练，调整可训练参数。"},
        "bad": {"action": "tool", "name": "calculate",
                "arguments": {"expression": "1 + 1"}},
        "reason": "这是概念问题，调用计算器不能回答它。",
    },
]


def schema_errors(value):
    """检查结构；结构正确不代表动作符合用户需求。"""
    if not isinstance(value, dict):
        return ["输出必须是 JSON 对象"]
    action = value.get("action")
    if action in ("clarify", "answer"):
        return [] if isinstance(value.get("content"), str) and value["content"].strip() else ["缺少有效 content"]
    if action != "tool":
        return ["未知 action"]
    argument_name = {"calculate": "expression", "get_weather": "city"}.get(value.get("name"))
    if argument_name is None:
        return ["未知工具"]
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        return ["arguments 必须是对象"]
    if not isinstance(arguments.get(argument_name), str) or not arguments[argument_name].strip():
        return ["缺少有效参数：" + argument_name]
    return []


def action_matches(actual, expected):
    """入门检查：动作、工具及参数。回答文本质量仍需单独评测。"""
    if schema_errors(actual) or actual["action"] != expected["action"]:
        return False
    if expected["action"] == "tool":
        return actual["name"] == expected["name"] and actual["arguments"] == expected["arguments"]
    return True


def training_record(case):
    # assistant.content 是模型应学习生成的 JSON 文本，不是工具执行结果。
    return {"messages": [
        {"role": "system", "content": RULES},
        {"role": "user", "content": case["user"]},
        {"role": "assistant", "content": json.dumps(case["expected"], ensure_ascii=False)},
    ]}


def main():
    print("步骤 1 | 明确规则：计算调用工具；缺城市则追问；概念问题直接回答。")
    print("步骤 2 | 展示人工标注的正确动作：")
    for case in CASES:
        print("  用户：", case["user"])
        print("  标签：", json.dumps(case["expected"], ensure_ascii=False))

    print("步骤 3 | 检查固定错误示范（不是模型预测）：")
    results = []
    for case in CASES:
        structure_ok = not schema_errors(case["bad"])
        action_ok = action_matches(case["bad"], case["expected"])
        print("  {} | 结构={} | 动作={} | {}".format(
            case["id"], "通过" if structure_ok else "失败",
            "通过" if action_ok else "失败", case["reason"]))
        results.append({"id": case["id"], "structure_ok": structure_ok,
                        "action_ok": action_ok, "reason": case["reason"]})

    output = Path(__file__).absolute().parent
    samples = output / "lesson01.samples.jsonl"
    records = [training_record(case) for case in CASES]
    samples.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    report = output / "lesson01.report.json"
    report.write_text(json.dumps({"kind": "handcrafted_demonstration",
                                  "model_called": False, "training_run": False,
                                  "sample_count": len(records), "checks": results},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("步骤 4 | 已保存 3 条教学样本：", samples)
    print("检查报告：", report)
    print("结果：三个错误都能通过结构检查，但都未通过动作检查。")
    print("边界：本课没有调用或训练模型；3 条样本仅用于理解数据格式。")


if __name__ == "__main__":
    main()
