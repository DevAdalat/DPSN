import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import itertools


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name,
        dataset_config=None,
        split="train",
        text_column="text",
        seq_len=1024,
        buffer_size=10000,
        streaming=True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_column = text_column
        self.buffer_size = buffer_size
        self.streaming = streaming

        # Load dataset initially to check validity, but actual iterator is created in __iter__
        # to support multiprocessing
        self._dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=streaming
        )

    def __iter__(self):
        # Handle Multiprocessing Workload Distribution
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            iterator = iter(self._dataset)
        else:
            # Multiple processes: split the dataset
            # Hugging Face streaming datasets support sharding
            # We assume the dataset supports .shard() or we just rely on the iterator being unique
            # Ideally we shard.
            # Note: Not all streaming datasets support efficient sharding.
            # We try to use the recommended sharding approach.
            try:
                # Shard the dataset for this worker
                sharded_dataset = self._dataset.shard(
                    num_shards=worker_info.num_workers, index=worker_info.id
                )
                iterator = iter(sharded_dataset)
            except AttributeError:
                # Fallback if shard is not available (though it should be on IterableDataset)
                # We skip examples: start at worker_id, step by num_workers
                # This is less efficient for streaming but ensures unique data
                iterator = itertools.islice(
                    iter(self._dataset), worker_info.id, None, worker_info.num_workers
                )

        more_examples = True

        while more_examples:
            buffer = []
            try:
                for _ in range(self.buffer_size):
                    buffer.append(next(iterator)[self.text_column])
            except StopIteration:
                more_examples = False

            if not buffer:
                break

            tokenized_buffer = self.tokenizer(
                buffer,
                truncation=True,
                max_length=self.seq_len * 8,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            # Flatten
            all_tokens = list(itertools.chain(*tokenized_buffer))

            # Yield chunks of seq_len
            for i in range(0, len(all_tokens) - self.seq_len, self.seq_len):
                chunk = all_tokens[i : i + self.seq_len + 1]  # +1 for target
                if len(chunk) == self.seq_len + 1:
                    yield torch.tensor(chunk, dtype=torch.long)


def get_streaming_dataloader(
    tokenizer_name,
    dataset_name,
    dataset_config=None,
    split="train",
    text_column="text",
    seq_len=1024,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
        seq_len=seq_len,
    )

    # Configure DataLoader for performance
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataloader, tokenizer.vocab_size
