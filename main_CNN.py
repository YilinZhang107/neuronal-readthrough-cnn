"""S2 CNN and original sequence encoder, adapted from the research scripts.

The classification head has one hidden layer: 4096 -> 256 -> 1.
Training is implemented in train.py.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def custom_pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    Pads and truncates a list of sequences (integer-encoded).
    
    Args:
        sequences (list of list of int): List of sequences.
        maxlen (int): Maximum length of all sequences.
        padding (str): 'pre' or 'post', pad either before or after each sequence.
        truncating (str): 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence.
        value (int): Value to pad the sequences to the desired length. This will typically be 0.

    Returns:
        np.ndarray: Numpy array with shape (len(sequences), maxlen).
    """
    padded_data = np.full((len(sequences), maxlen), value, dtype=np.int32) 
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue 
        
        if len(seq) > maxlen:
            if truncating == 'pre':
                trunc = seq[-maxlen:]
            elif truncating == 'post':
                trunc = seq[:maxlen]
            else:
                raise ValueError(f"Truncating type '{truncating}' not understood. Choose 'pre' or 'post'.")
            padded_data[i, :] = trunc
        else:
            if padding == 'post':
                padded_data[i, :len(seq)] = seq
            elif padding == 'pre':
                padded_data[i, -len(seq):] = seq
            else:
                raise ValueError(f"Padding type '{padding}' not understood. Choose 'pre' or 'post'.")
    return padded_data


class GeneSequenceDataset(Dataset):
    def __init__(self, sequences, labels, vocab_map, max_len):
        self.sequences = sequences
        self.labels = labels
        self.vocab_map = vocab_map
        self.max_len = max_len
        
        if not self.vocab_map: # Handle empty vocabulary
            print("错误: GeneSequenceDataset 初始化时词汇表为空。")
            self.num_classes_vocab = 0 
            self.padding_value_index = 0 # Default, but meaningless with empty vocab
            self.unknown_char_index = 0
        else:
            self.num_classes_vocab = len(vocab_map)
            self.padding_value_index = 0 
            self.unknown_char_index = 0


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.num_classes_vocab == 0:
            print(f"警告: 尝试从词汇表为空的Dataset获取项目。序列: '{self.sequences[idx]}'")
            dummy_sequence_tensor = torch.zeros((0, self.max_len), dtype=torch.float32)
            dummy_label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return dummy_sequence_tensor, dummy_label_tensor

        seq = self.sequences[idx]
        label = self.labels[idx]

        # 整数编码
        integer_encoded = [self.vocab_map.get(char, self.unknown_char_index) for char in seq]
        
        padded_sequence_np = custom_pad_sequences(
            [integer_encoded], 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post', 
            value=self.padding_value_index 
        )[0] 

        one_hot_sequence = np.zeros((self.max_len, self.num_classes_vocab), dtype=np.float32)
        for i, char_idx in enumerate(padded_sequence_np):
            if i < self.max_len : 
                one_hot_sequence[i, char_idx] = 1.0
        
        sequence_tensor = torch.tensor(one_hot_sequence, dtype=torch.float32).permute(1, 0) 
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return sequence_tensor, label_tensor


class CNN1DModel(nn.Module):
    def __init__(self, vocab_size, max_len, num_filters1=64, kernel_size1=7, pool_size1=3, dropout1=0.3,
                 num_filters2=128, kernel_size2=5, pool_size2=2, dropout2=0.4,
                 num_filters3=256, kernel_size3=3, pool_size3=2, dropout3=0.5,
                 dense_units1=256, dense_dropout1=0.5):
        super(CNN1DModel, self).__init__()
        
        if vocab_size == 0:
            raise ValueError("Vocab size cannot be 0. Check if any valid characters (ATCGU) were found in the data.")

        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=num_filters1, kernel_size=kernel_size1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size1)
        self.drop1 = nn.Dropout(dropout1)

        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size2, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size2)
        self.drop2 = nn.Dropout(dropout2)

        self.conv3 = nn.Conv1d(in_channels=num_filters2, out_channels=num_filters3, kernel_size=kernel_size3, padding='same')
        self.bn3 = nn.BatchNorm1d(num_filters3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=pool_size3)
        self.drop3 = nn.Dropout(dropout3)

        self.flatten = nn.Flatten()
        
        # Preserve the original shape-probe order and RNG consumption.
        with torch.no_grad():
            dummy_input = torch.zeros(1, vocab_size, max_len)
            dummy_out = self.pool3(self.drop3(self.relu3(self.bn3(self.conv3(
                            self.pool2(self.drop2(self.relu2(self.bn2(self.conv2(
                            self.pool1(self.drop1(self.relu1(self.bn1(self.conv1(dummy_input)))))))))))))))
            flattened_size = dummy_out.flatten(1).shape[1]

        self.fc1 = nn.Linear(flattened_size, dense_units1)
        self.bn_fc1 = nn.BatchNorm1d(dense_units1)
        self.relu_fc1 = nn.ReLU()
        self.drop_fc1 = nn.Dropout(dense_dropout1)

        self.fc_out = nn.Linear(dense_units1, 1)


    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.flatten(x)
        x = self.drop_fc1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    from train import main
    main()
