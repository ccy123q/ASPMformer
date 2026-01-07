from .ASPMFormer import ASPMFormer


def model_select(name):
    # 统一将模型名称转为大写
    name = name.upper()
    print(f"Received model name: {name}")
    if name == "ASPMFORMER":
        return ASPMFormer
    else:
        raise NotImplementedError(f"Model {name} not implemented.")

