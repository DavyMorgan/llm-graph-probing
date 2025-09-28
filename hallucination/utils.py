import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def test_fn(model, data_loader, device, num_layers):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data in data_loader:
            if num_layers > 0:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = out.argmax(dim=-1)
                all_preds.append(pred.cpu())
                all_labels.append(data.y.cpu())
                all_outputs.append(out.cpu())
            else:
                x, y = data
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=-1)
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())
                all_outputs.append(out.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, precision, recall, f1, cm
