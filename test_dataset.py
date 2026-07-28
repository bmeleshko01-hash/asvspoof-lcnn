from xml.parsers.expat import model
from torch import nn

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pre_commit import output
from src.model import BaselineModel


from src.datasets.collate import collate_fn


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="baseline",
)
def main(config: DictConfig):
    print("Конфиг успешно загружен.")

    train_dataset = instantiate(
        config.datasets.train,
        instance_transforms=config.transforms.instance_transforms.train,
    )

    print("Размер датасета:", len(train_dataset))

    item = train_dataset[0]

    print("Ключи одного объекта:", item.keys())
    print("Размер data_object:", item["data_object"].shape)
    print("Метка:", item["labels"])

    train_dataloader = instantiate(
        config.dataloader,
        dataset=train_dataset,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_dataloader))

    print("Ключи батча:", batch.keys())
    print("Размер data_object батча:", batch["data_object"].shape)
    print("Размер labels батча:", batch["labels"].shape)
    print("Метки батча:", batch["labels"])

    model = BaselineModel()

    output = model(data_object=batch["data_object"])

    print(output["logits"].shape)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        output["logits"],
        batch["labels"],
    )

    print(loss)


if __name__ == "__main__":
    main()