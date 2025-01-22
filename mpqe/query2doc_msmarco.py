# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MPQE: LLM generations from ms-marco queries"""


import json
import os
import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{Wang2023Query2docQE,
  title={Query2doc: Query Expansion with Large Language Models},
  author={Liang Wang and Nan Yang and Furu Wei},
  year={2023}
}
"""

# You can copy an official description
_DESCRIPTION = """\
This dataset contains GPT-3.5 (text-davinci-003) generations from MS-MARCO queries.
"""

_URLS = {
    "train": "train.jsonl",
    "dev": "dev.jsonl",
    "test": "test.jsonl",
    "trec_dl2019": "trec_dl2019.jsonl",
    "trec_dl2020": "trec_dl2020.jsonl",
}


class Query2docMsmarco(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='plain_text', version=VERSION, description='plain text')
    ]

    def _info(self):
        features = datasets.Features(
            {
                "query_id": datasets.Value("string"),
                "query": datasets.Value("string"),
                "pseudo_doc": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(_URLS)
        print(downloaded_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name="trec_dl2019",
                gen_kwargs={
                    "filepath": downloaded_files["trec_dl2019"],
                    "split": "trec_dl2019"
                },
            ),
            datasets.SplitGenerator(
                name="trec_dl2020",
                gen_kwargs={
                    "filepath": downloaded_files["trec_dl2020"],
                    "split": "trec_dl2020"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                # Yields examples as (key, example) tuples
                yield key, {
                    "query_id": data["query_id"],
                    "query": data["query"],
                    "pseudo_doc": data['pseudo_doc']
                }
