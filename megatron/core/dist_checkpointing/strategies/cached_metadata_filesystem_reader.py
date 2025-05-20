# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
from siflowai import sifile 
""" FS Reader with metadata cached support. """
import collections
import torch
import dataclasses
import io
import operator
import pickle
import queue
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Union
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    IO,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, Metadata
from torch import Tensor
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.futures import Future
class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Attributes:
        _cached_metadata (Metadata or None): Cached metadata from the file system.
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cached_metadata = None

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cached_metadata is None:
            self._cached_metadata = super().read_metadata()
        return self._cached_metadata
class UbiCachedMetadataFileSystemReader(CachedMetadataFileSystemReader):
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:

        per_file: Dict[str, List] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            si = sifile(str(new_path))  
            try:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    offset = item_md.offset
                    length = item_md.length

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = si.read(offset, length)
                        byte_stream = io.BytesIO(read_bytes)
                        byte_stream.seek(0)
                        planner.load_bytes(req, byte_stream)
                    else:
                        read_bytes = si.read(offset, length)
                        # 使用 numpy 进行反序列化，再转成 torch tensor
                        dtype = req.properties.dtype
                        shape = tuple(req.lengths)
                        np_dtype = np.dtype(str(dtype).replace('torch.', ''))
                        np_arr = np.frombuffer(read_bytes, dtype=np_dtype).reshape(shape)
                        tensor = torch.from_numpy(np_arr)

                        # 处理偏移 (optional，根据是否分片)
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert target_tensor.size() == tensor.size(), (
                            f"Tensor size mismatch at index {req.storage_index}: {target_tensor.size()} vs {tensor.size()}"
                        )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)
            finally:
                si.close()

        fut = Future()
        fut.set_result(None)
        return fut