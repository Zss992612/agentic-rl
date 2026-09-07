from tau2.domains.retail.environment import get_environment, get_tasks


def test_retail_environment_loads() -> None:
    tasks = get_tasks("train")
    environment = get_environment()
    tool_names = {tool.name for tool in environment.get_tools()}

    assert tasks
    assert environment.domain_name == "retail"
    assert "get_order_details" in tool_names
