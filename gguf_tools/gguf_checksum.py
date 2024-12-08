#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from hashlib import sha256, _Hash # TODO Figure out of there is another way to type hint this.
import json
from typing import Any, Callable


from gguf.gguf_reader import GGUFReader, ReaderField, ReaderTensor
from logger.logger import Logger
logger = Logger(logger_name=__name__)


from dataclasses import dataclass

@dataclass
class GgufChecksumDataClass:

    def __post_init__(self):
        self._metadata_hash: _Hash = sha256()
        self._kv_hash: _Hash = sha256()
        self._tensor_info_hash: _Hash = sha256()
        self._tensor_data_hash: _Hash = sha256()
        self._overall_hash: _Hash = sha256()

        self._curr_kv_hash: _Hash = None
        self._result_dict: dict[str, Any] = {}

    @property
    def result_dict(self) -> dict[str, Any]:
        return self._result_dict

    @result_dict.setter
    def result_dict(self, tuple_: tuple[Any,...]) -> None:
        if len(tuple_) == 2:
            try:
                key, value = tuple_
            except ValueError:
                raise ValueError("Pass a tuple with two items")
            else:
                self._result_dict[key] = value
        elif len(tuple) == 3:
            try:
                key, sub_key, value = tuple_
            except ValueError:
                raise ValueError("Pass a tuple with two items")
            else:
                self._result_dict[key][sub_key] = value
        else:
            raise ValueError("Pass a tuple with two or three items.")

    @property
    def metadata_hash(self) -> sha256:
        return self._metadata_hash

    @metadata_hash.setter
    def metadata_hash(self, value: sha256|bytes) -> None:
        self._metadata_hash.update(value)

    @property
    def kv_hash(self) -> _Hash:
        return self._kv_hash

    @kv_hash.setter
    def kv_hash(self, value: sha256|bytes) -> None:
        self._kv_hash.update(value)

    @property
    def tensor_info_hash(self) -> _Hash:
        return self._tensor_info_hash

    @tensor_info_hash.setter
    def tensor_info_hash(self, value: sha256|bytes) -> None:
        self._tensor_info_hash.update(value)

    @property
    def tensor_data_hash(self) -> _Hash:
        return self._tensor_data_hash

    @tensor_data_hash.setter
    def tensor_data_hash(self, value: sha256|bytes) -> None:
        self._tensor_data_hash.update(value)

    @property
    def overall_hash(self) -> _Hash:
        return self._overall_hash

    @overall_hash.setter
    def overall_hash(self, value: sha256|bytes) -> None:
        self._overall_hash.update(value)


from config.file_specific_configs import FileSpecificConfigs

from config.config import OUTPUT_FOLDER, INPUT_FOLDER
from config.file_specific_configs import FileSpecificConfigs
config: Callable = FileSpecificConfigs().config


class GgufChecksum:

    def __init__(self, **kwargs):
        _gguf_reader: str = kwargs.pop("model") 
        self.gguf_reader: GGUFReader = GGUFReader(_gguf_reader, "r") # Read in the model
        self.gguf_checksum_dataclass: GgufChecksumDataClass = GgufChecksumDataClass() # Create SH256 hash variables
        self.hash_individual_kvs: bool = kwargs.pop("hash_individual_kvs") 
        self.hash_individual_tensors: bool = kwargs.pop("hash_individual_tensors")
        self.skip_kvs: bool = kwargs.pop("skip_kvs")
        self.skip_tensors: bool = kwargs.pop("skip_tensors")
        self.skip_tensor_data: bool = kwargs.pop("skip_tensor_data")
        self.json: bool = kwargs.pop("json")
        self.output_file: str = kwargs.pop("output_file")

    @staticmethod
    def kv_sortfun(field: ReaderField) -> str:
        return field.name

    @staticmethod
    def tensor_sortfun(tensor: ReaderTensor) -> str:
        return tensor.name

    def gguf_checksum(self) -> None:

        # Create a sub-dictionary for hashes of KVS if they're in the args
        if self.hash_individual_kvs:
            self.gguf_checksum_dataclass.result_dict = ("kvs", {})

        # Create a sub-dictionary for hashes of the individual tensors if they're in the args.
        if self.hash_individual_tensors:
            self.gguf_checksum_dataclass.result_dict = ("tensors", {})

        if not self.skip_kvs:
            self.get_kvs_metadata()

        if not self.skip_tensors:
            self.get_tensor_metadata()

        self.set_kvs_metadata()
        self.set_tensor_metadata()
        self.set_metadata_hash()
        self.set_tensor_data_hash()
        self.set_overall_hash_then_save_to_json()
        return

    def get_kvs_metadata(self) -> None:

        # Sort the KVs by field name
        kvs = sorted(self.gguf_reader.fields.values(), key=self.kv_sortfun)

        for kvs_field in kvs:

            # Skip the spare field in KVS
            if kvs_field.name.startswith("__SPARE"):
                continue

            curr_kv_hash = sha256()

            for part in kvs_field.parts:
                pb = part.tobytes()
                self.gguf_checksum_dataclass.metadata_hash = pb
                self.gguf_checksum_dataclass.kv_hash = pb

                if self.hash_individual_kvs:
                    curr_kv_hash.update(pb)

                self.gguf_checksum_dataclass.overall_hash = pb

            if self.hash_individual_kvs:
                if self.json:
                    self.gguf_checksum_dataclass.result_dict = ("kvs", kvs_field.name, curr_kv_hash.hexdigest())
                else:
                    logger.info(f"HASH {'KV':15} {curr_kv_hash.hexdigest()} {kvs_field.name!r}")

    def get_tensor_metadata(self) -> None:

        tis = sorted(self.gguf_reader.tensors, key=self.tensor_sortfun)

        for tensor in tis:
            # Skip the offset
            for part in tensor.field.parts[:-1]:
                pb = part.tobytes()

                self.gguf_checksum_dataclass.metadata_hash = pb
                self.gguf_checksum_dataclass.tensor_info_hash = pb
                self.gguf_checksum_dataclass.overall_hash = pb

        if not self.skip_tensor_data:
            for tensor in tis:
                tb = tensor.data.tobytes()
                self.gguf_checksum_dataclass.tensor_data_hash = tb
                self.gguf_checksum_dataclass.overall_hash = tb

                if self.hash_individual_tensors:
                    tensor_hash = sha256()
                    tensor_hash.update(tb)

                    if self.json:
                        self.gguf_checksum_dataclass.result_dict = ("tensors", tensor.name, tensor_hash.hexdigest())
                    else:
                        logger.info(
                            f"HASH {'TENSOR':15} {tensor_hash.hexdigest()} {tensor.name!r}",
                        )

                elif not self.json:
                    logger.info(
                        f"HASH {'RUNNING':15} {self.gguf_checksum_dataclass.overall_hash.hexdigest()} {tensor.name!r}",
                    )

    def set_kvs_metadata(self) -> None:
        if not self.skip_kvs:
            if self.json:
                self.gguf_checksum_dataclass.result_dict = ("kv_metadata", self.gguf_checksum_dataclass.kv_hash.hexdigest())
            else:
                logger.info(f"HASH {'KV_METADATA':15} {self.gguf_checksum_dataclass.kv_hash.hexdigest()}")
        return

    def set_tensor_metadata(self) -> None:
        if not self.skip_tensors:
            if self.json:
                self.gguf_checksum_dataclass.result_dict = ("tensor_metadata", self.gguf_checksum_dataclass.tensor_info_hash.hexdigest())
                if not self.skip_tensor_data:
                    self.gguf_checksum_dataclass.result_dict = ("tensor_data", self.gguf_checksum_dataclass.tensor_data_hash.hexdigest())
            else:
                logger.info(f"HASH {'TENSOR_METADATA':15} {self.gguf_checksum_dataclass.tensor_info_hash.hexdigest()}")

    def set_metadata_hash(self) -> None:
        if not (self.skip_kvs and self.skip_tensors):
            if self.json:
                self.gguf_checksum_dataclass.result_dict = ("metadata", self.gguf_checksum_dataclass.metadata_hash.hexdigest())
            else:
                logger.info(f"HASH {'METADATA':15} {self.gguf_checksum_dataclass.metadata_hash.hexdigest()}")

    def set_tensor_data_hash(self) -> None:
        if not (self.skip_tensors or self.skip_tensor_data):
            if self.json:
                self.gguf_checksum_dataclass.result_dict = ("tensor_data", self.gguf_checksum_dataclass.tensor_data_hash.hexdigest())
            else:
                logger.info(f"HASH {'TENSOR_DATA':15} {self.gguf_checksum_dataclass.tensor_data_hash.hexdigest()}")

    def set_overall_hash_then_save_to_json(self) -> None:
        if self.json:
            self.gguf_checksum_dataclass.result_dict = ("overall", self.gguf_checksum_dataclass.overall_hash.hexdigest())
            json.dump(self.gguf_checksum_dataclass.result_dict , sys.stdout)
        else:
            logger.info(f"HASH {'OVERALL':15} {self.gguf_checksum_dataclass.overall_hash.hexdigest()}")







def gguf_checksum(args: argparse.Namespace) -> None:

    # Read in the model
    reader = GGUFReader(args.model, "r")

    # Create SH256 hash variables
    metadata_hash = sha256()
    kv_hash = sha256()
    tensor_info_hash = sha256()
    tensor_data_hash = sha256()
    overall_hash = sha256()

    # Create a result dictionary
    result: dict[str, Any] = {}

    # Create a sub-dictionary for hashes of KVS if they're in the args
    if args.hash_individual_kvs:
        result["kvs"] = {}

    # Create a sub-dictionary for hashes of the individual tensors if they're in the args.
    if args.hash_individual_tensors:
        result["tensors"] = {}

    if not args.skip_kvs:

        def kv_sortfun(field: ReaderField) -> str:
            return field.name

        kvs = sorted(reader.fields.values(), key=kv_sortfun)

        for field in kvs:

            # Skip the spare field in KVS
            if field.name.startswith("__SPARE"):
                continue

            curr_kv_hash = sha256()

            for part in field.parts:
                pb = part.tobytes()
                metadata_hash.update(pb)
                kv_hash.update(pb)

                if args.hash_individual_kvs:
                    curr_kv_hash.update(pb)

                overall_hash.update(pb)

            if args.hash_individual_kvs:
                if args.json:
                    result["kvs"][field.name] = curr_kv_hash.hexdigest()
                else:
                    logger.info(f"HASH {'KV':15} {curr_kv_hash.hexdigest()} {field.name!r}")

    if not args.skip_tensors:

        def tensor_sortfun(tensor: ReaderTensor) -> str:
            return tensor.name

        tis = sorted(reader.tensors, key=tensor_sortfun)
        for tensor in tis:
            # Skip the offset
            for part in tensor.field.parts[:-1]:
                pb = part.tobytes()
                metadata_hash.update(pb)
                tensor_info_hash.update(pb)
                overall_hash.update(pb)

        if not args.skip_tensor_data:
            for tensor in tis:
                tb = tensor.data.tobytes()
                tensor_data_hash.update(tb)
                overall_hash.update(tb)

                if args.hash_individual_tensors:
                    tensor_hash = sha256()
                    tensor_hash.update(tb)

                    if args.json:
                        result["tensors"][tensor.name] = tensor_hash.hexdigest()
                    else:
                        logger.info(
                            f"HASH {'TENSOR':15} {tensor_hash.hexdigest()} {tensor.name!r}",
                        )

                elif not args.json:
                    logger.info(
                        f"HASH {'RUNNING':15} {overall_hash.hexdigest()} {tensor.name!r}",
                    )

    if not args.skip_kvs:
        if args.json:
            result["kv_metadata"] = kv_hash.hexdigest()
        else:
            logger.info(f"HASH {'KV_METADATA':15} {kv_hash.hexdigest()}")

    if not args.skip_tensors:
        if args.json:
            result["tensor_metadata"] = tensor_info_hash.hexdigest()
            if not args.skip_tensor_data:
                result["tensor_data"] = tensor_data_hash.hexdigest()
        else:
            logger.info(f"HASH {'TENSOR_METADATA':15} {tensor_info_hash.hexdigest()}")

    if not (args.skip_kvs and args.skip_tensors):
        if args.json:
            result["metadata"] = metadata_hash.hexdigest()
        else:
            logger.info(f"HASH {'METADATA':15} {metadata_hash.hexdigest()}")

    if not (args.skip_tensors or args.skip_tensor_data):
        if args.json:
            result["tensor_data"] = tensor_data_hash.hexdigest()
        else:
            logger.info(f"HASH {'TENSOR_DATA':15} {tensor_data_hash.hexdigest()}")

    if args.json:
        result["overall"] = overall_hash.hexdigest()
        json.dump(result, sys.stdout)
    else:
        logger.info(f"HASH {'OVERALL':15} {overall_hash.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checksum utility for GGUF files")
    parser.add_argument(
        "model",
        type=str,
        help="GGUF format model filename",
    )
    parser.add_argument(
        "--skip-kvs",
        action="store_true",
        help="Skip KV metadata in result",
    )
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="Skip tensors in result, implies --skip-tensor-data as well",
    )
    parser.add_argument(
        "--skip-tensor-data",
        action="store_true",
        help="Skip tensor data in result",
    )
    parser.add_argument(
        "--hash-individual-kvs",
        action="store_true",
        help="Include individual KV hashes in result, no effect when --skip-kvs specified",
    )
    parser.add_argument(
        "--hash-individual-tensors",
        action="store_true",
        help="Include individual tensor data hashes in result, no effect when --skip-tensors or --skip-tensor-data specified",
    )

    parser.add_argument("--json", action="store_true", help="Produce JSON output")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    gguf_checksum(args)


if __name__ == "__main__":
    main()
