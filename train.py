import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate

from datasets import load_dataset
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score

from loguru import logger
import fire

from model import get_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
id2label = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}
label2id = {"POSITIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2}


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader, df_eval):
    y_pred = []
    y_true = []
    metric = evaluate.load("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        for i in predictions:
            y_pred.append(i)

        labels = batch["labels"].cpu().numpy()
        y_true.extend(labels)
        # print("after predictions")

        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)

    accuracy = metric.compute()["accuracy"]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    cls_report = classification_report(
        y_true, y_pred, target_names=["positive", "negative", "neutral"]
    )
    print(cls_report)
    ConfusionMatrixDisplay(cm).plot()
    df = df_eval.copy()
    df["predictions"] = y_pred

    # Turn class ids into class labels
    df["class"] = df["predictions"].map(id2label)
    df.to_csv("clinincal-bigbird-1.csv")


def load_data(model_id):
    df = pd.read_csv("client1.csv")
    df2 = pd.read_csv("test.csv")

    X_train = df["text"]
    y_train = df["label"]
    X_test = df2["text"]
    y_test = df2["label"]
    df_eval = pd.DataFrame(X_test, columns=["text"])
    df_eval["label"] = y_test

    d = {
        "train": Dataset.from_dict({"label": y_train, "text": X_train}),
        "test": Dataset.from_dict({"label": y_test, "text": X_test}),
    }
    d = DatasetDict(d)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )  # "yikuan8/Clinical-BigBird"

    def _tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = d.map(_tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
    )

    return trainloader, testloader, df_eval


def fit(trainloader, model_id, save_dir, use_peft, peft_method, num_labels, epochs):
    LEARNING_RATE = 2e-5

    net = get_model(model_id, use_peft, peft_method, num_labels, id2label, label2id)
    net = net.to(DEVICE)
    # Define optimizer and loss function
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_samples = 0
        predictions = []
        true_labels = []

        for batch in trainloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = net(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += len(labels)

            # Store predictions and true labels for evaluation
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # Calculate metrics after each epoch
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="weighted")
        recall = recall_score(true_labels, predictions, average="weighted")
        precision = precision_score(true_labels, predictions, average="weighted")

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {total_loss / total_samples:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print("-" * 50)

    # Save the trained model if needed
    net.save_pretrained(save_dir)
    return net


def evaluate_model(net, test_loader, device):
    net.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = net(input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    precision = precision_score(true_labels, predictions, average="weighted")

    return (
        true_labels,
        predictions,
        {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "precision": precision,
        },
    )


def main(
    model_id: str,
    save_dir: str,
    num_labels: int,
    epochs: int,
    use_peft: bool,
    peft_method: str = None,
):
    trainloader, testloader, df_eval = load_data(model_id)
    net = fit(
        trainloader, model_id, save_dir, use_peft, peft_method, num_labels, epochs
    )
    test_metrics = evaluate_model(net, testloader, DEVICE)
    logger.info(f"Test Metrics: {test_metrics[-1]}")


if __name__ == "__main__":
    fire.Fire(main)
