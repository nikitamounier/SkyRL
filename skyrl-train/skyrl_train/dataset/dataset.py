import datasets
from loguru import logger
import os
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.modalities import normalize_modalities_config, plan_modalities_for_batch
from skyrl_train.modalities.types import SampleModalityData
from skyrl_train.modalities.batching import populate_sample_occurrences


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        modalities_config: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers
        self.modality_specs = normalize_modalities_config(modalities_config)

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        loaded_datasets = []
        for source in self.datasets:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".parquet":
                ds = datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
            elif ext in [".json", ".jsonl"]:
                ds = datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            else:
                # Treat as HF dataset spec: "name" or "name:split"
                dataset_name, has_split, split = source.partition(":")
                try:
                    ds_dict = datasets.load_dataset(path=dataset_name, keep_in_memory=True)
                except ValueError:
                    raise ValueError(f"Dataset `{dataset_name}` not found on Hugging Face.")
                split = split if has_split else "train"
                if split not in ds_dict:
                    raise ValueError(
                        f"Split `{split}` not found in dataset `{dataset_name}`. Configured split was `{split}` and default is `train`"
                    )
                ds = ds_dict[split]
            loaded_datasets.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded_datasets)

        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: Dict[str, Any] = dict(self.dataframe[item])
        modalities_payload = self._extract_modalities(row_dict)
        modality_metadata = SampleModalityData(payloads=modalities_payload)

        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}
        extra["modalities"] = modality_metadata
        uid = str(item)

        return messages, env_class, extra, uid

    def collate_fn(self, item_list):
        prompts = [prompt for prompt, _, _, _ in item_list]
        modality_payloads = []
        modality_entries: List[SampleModalityData] = []
        for _, _, env_extras, _ in item_list:
            sample_modalities: SampleModalityData = env_extras.get("modalities", SampleModalityData())
            modality_entries.append(sample_modalities)
            modality_payloads.append(sample_modalities.payloads)
        modality_plans = plan_modalities_for_batch(prompts, modality_payloads, self.modality_specs)

        all_inputs = []
        for (prompt, env_class, env_extras, item_uids), plan, sample_modalities in zip(
            item_list, modality_plans, modality_entries
        ):
            env_extras = dict(env_extras)
            modalities_data: SampleModalityData = env_extras.get("modalities", SampleModalityData())
            if modalities_data is sample_modalities:
                modalities_data = modalities_data.clone()
            modalities_data.plans = plan
            populate_sample_occurrences(modalities_data)
            env_extras["modalities"] = modalities_data
            all_inputs.append(
                {
                    "prompt": prompt,
                    "env_class": env_class,
                    "env_extras": env_extras,
                    "uid": item_uids,
                }
            )
        return all_inputs

    def __len__(self):
        return len(self.dataframe)

    def _extract_modalities(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Collect modality payloads from the dataset row."""
        if not self.modality_specs:
            return {}

        modalities_payload: Dict[str, Any] = {}

        # Preferred schema: nested under "modalities"
        nested_modalities = row_dict.get("modalities")
        if isinstance(nested_modalities, dict):
            for modality_id in self.modality_specs:
                if modality_id in nested_modalities:
                    modalities_payload[modality_id] = nested_modalities[modality_id]
            # remove nested dict to avoid leaking into extras twice
            row_dict.pop("modalities", None)

        # Fallback schema: top-level keys
        for modality_id in self.modality_specs:
            if modality_id in row_dict:
                modalities_payload.setdefault(modality_id, row_dict.pop(modality_id))

        return modalities_payload
