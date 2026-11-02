from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import queue
import hashlib
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timedelta


class DataSourceType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    STREAM = "stream"
    DATABASE = "database"
    API = "api"
    KAFKA = "kafka"
    MQTT = "mqtt"
    ROS = "ros"
    ROS2 = "ros2"
    WEBSOCKET = "websocket"
    HTTP = "http"


class DataFormat(Enum):
    RAW = "raw"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    NPY = "npy"
    NPZ = "npz"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    BAG = "bag"
    MCAP = "mcap"
    PCD = "pcd"
    PLY = "ply"
    ROS_BAG = "ros_bag"
    CUSTOM = "custom"


class IngestionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DataSource:
    id: str
    source_type: DataSourceType
    format: DataFormat
    path: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class IngestionJob:
    id: str
    source_id: str
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataRecord:
    id: str
    source_id: str
    timestamp: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    checksum: Optional[str] = None


class DataSchema:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.version = schema.get('version', '1.0')
        self.fields = schema.get('fields', {})
        self.required = schema.get('required', [])
        self.types = schema.get('types', {})

    def validate(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []

        for field in self.required:
            if field not in record:
                errors.append(f"Missing required field: {field}")

        for field, expected_type in self.types.items():
            if field in record:
                value = record[field]
                if not self._check_type(value, expected_type):
                    errors.append(f"Field {field} has incorrect type: expected {expected_type}")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


class BaseReader(ABC):
    @abstractmethod
    def read(self, path: str, **kwargs) -> List[DataRecord]:
        pass

    @abstractmethod
    def stream(self, path: str, batch_size: int = 1000) -> Any:
        pass


class FileReader(BaseReader):
    def __init__(self, format: DataFormat, schema: Optional[DataSchema] = None):
        self.format = format
        self.schema = schema

    def read(self, path: str, **kwargs) -> List[DataRecord]:
        records = []

        if self.format == DataFormat.JSON:
            records = self._read_json(path)
        elif self.format == DataFormat.CSV:
            records = self._read_csv(path)
        elif self.format == DataFormat.NPY:
            records = self._read_npy(path)
        elif self.format == DataFormat.NPZ:
            records = self._read_npz(path)
        elif self.format == DataFormat.PARQUET:
            records = self._read_parquet(path)
        elif self.format in [DataFormat.PNG, DataFormat.JPEG, DataFormat.TIFF]:
            records = self._read_image(path)

        if self.schema:
            validated = []
            for record in records:
                valid, _ = self.schema.validate(record.data if isinstance(record.data, dict) else {'data': record.data})
                if valid:
                    validated.append(record)
            records = validated

        return records

    def stream(self, path: str, batch_size: int = 1000):
        if self.format == DataFormat.JSON:
            return self._stream_json(path, batch_size)
        elif self.format == DataFormat.CSV:
            return self._stream_csv(path, batch_size)
        return iter([])

    def _read_json(self, path: str) -> List[DataRecord]:
        with open(path, 'r') as f:
            data = json.load(f)

        records = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                records.append(DataRecord(
                    id=f"{path}_{i}",
                    source_id=path,
                    timestamp=time.time(),
                    data=item,
                    checksum=hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
                ))
        else:
            records.append(DataRecord(
                id=path,
                source_id=path,
                timestamp=time.time(),
                data=data,
                checksum=hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            ))

        return records

    def _read_csv(self, path: str) -> List[DataRecord]:
        import csv
        records = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                records.append(DataRecord(
                    id=f"{path}_{i}",
                    source_id=path,
                    timestamp=time.time(),
                    data=row,
                    checksum=hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()
                ))

        return records

    def _read_npy(self, path: str) -> List[DataRecord]:
        data = np.load(path)
        records = []

        if data.ndim == 0:
            records.append(DataRecord(
                id=path,
                source_id=path,
                timestamp=time.time(),
                data=data.item(),
                checksum=hashlib.md5(data.tobytes()).hexdigest()
            ))
        else:
            for i in range(min(len(data), 1000)):
                records.append(DataRecord(
                    id=f"{path}_{i}",
                    source_id=path,
                    timestamp=time.time(),
                    data=data[i],
                    checksum=hashlib.md5(data[i].tobytes()).hexdigest()
                ))

        return records

    def _read_npz(self, path: str) -> List[DataRecord]:
        data = np.load(path)
        records = []

        for key in data.files:
            arr = data[key]
            for i in range(min(len(arr), 1000)):
                records.append(DataRecord(
                    id=f"{path}_{key}_{i}",
                    source_id=path,
                    timestamp=time.time(),
                    data={'key': key, 'value': arr[i]},
                    checksum=hashlib.md5(arr[i].tobytes()).hexdigest()
                ))

        return records

    def _read_parquet(self, path: str) -> List[DataRecord]:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
            records = []

            for i, row in df.iterrows():
                records.append(DataRecord(
                    id=f"{path}_{i}",
                    source_id=path,
                    timestamp=time.time(),
                    data=row.to_dict(),
                    checksum=hashlib.md5(json.dumps(row.to_dict(), sort_keys=True).encode()).hexdigest()
                ))

            return records
        except ImportError:
            return []

    def _read_image(self, path: str) -> List[DataRecord]:
        image = cv2.imread(path)
        if image is not None:
            return [DataRecord(
                id=path,
                source_id=path,
                timestamp=time.time(),
                data=image,
                metadata={'width': image.shape[1], 'height': image.shape[0], 'channels': image.shape[2] if len(image.shape) > 2 else 1},
                checksum=hashlib.md5(image.tobytes()).hexdigest()
            )]
        return []

    def _stream_json(self, path: str, batch_size: int):
        return iter([])

    def _stream_csv(self, path: str, batch_size: int):
        import csv
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch


class DirectoryReader(BaseReader):
    def __init__(
        self,
        format: DataFormat,
        schema: Optional[DataSchema] = None,
        recursive: bool = True,
        pattern: str = "*"
    ):
        self.format = format
        self.schema = schema
        self.recursive = recursive
        self.pattern = pattern
        self.file_reader = FileReader(format, schema)

    def read(self, path: str, **kwargs) -> List[DataRecord]:
        records = []
        files = self._find_files(path)

        for file_path in files:
            try:
                file_records = self.file_reader.read(file_path)
                records.extend(file_records)
            except Exception:
                pass

        return records

    def stream(self, path: str, batch_size: int = 1000):
        files = self._find_files(path)
        for file_path in files:
            try:
                for batch in self.file_reader.stream(file_path, batch_size):
                    yield batch
            except Exception:
                pass

    def _find_files(self, path: str) -> List[str]:
        extensions = {
            DataFormat.JSON: 'json',
            DataFormat.CSV: 'csv',
            DataFormat.NPY: 'npy',
            DataFormat.NPZ: 'npz',
            DataFormat.PARQUET: 'parquet',
            DataFormat.PNG: 'png',
            DataFormat.JPEG: ['jpg', 'jpeg'],
            DataFormat.TIFF: ['tiff', 'tif'],
            DataFormat.PCD: 'pcd',
            DataFormat.PLY: 'ply'
        }

        ext = extensions.get(self.format, '*')
        if isinstance(ext, list):
            patterns = [os.path.join(path, f"**/*.{e}") for e in ext]
        else:
            patterns = [os.path.join(path, f"**/*.{ext}")]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=self.recursive))

        return sorted(files)


class StreamReader(BaseReader):
    def __init__(
        self,
        source_url: str,
        format: DataFormat,
        schema: Optional[DataSchema] = None,
        buffer_size: int = 1000
    ):
        self.source_url = source_url
        self.format = format
        self.schema = schema
        self.buffer_size = buffer_size
        self.running = False
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.thread: Optional[threading.Thread] = None

    def read(self, path: str, **kwargs) -> List[DataRecord]:
        return []

    def stream(self, path: str, batch_size: int = 1000):
        self.start()
        while self.running:
            try:
                batch = []
                for _ in range(batch_size):
                    item = self.buffer.get(timeout=1.0)
                    if item is None:
                        break
                    batch.append(item)
                if batch:
                    yield batch
            except queue.Empty:
                continue
        self.stop()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _run(self):
        pass


class DataTransformer:
    def __init__(self):
        self.transforms: List[Callable[[DataRecord], DataRecord]] = []

    def add_transform(self, transform: Callable[[DataRecord], DataRecord]):
        self.transforms.append(transform)

    def transform(self, record: DataRecord) -> DataRecord:
        for transform in self.transforms:
            record = transform(record)
        return record

    def transform_batch(self, records: List[DataRecord]) -> List[DataRecord]:
        return [self.transform(r) for r in records]


class DataValidator:
    def __init__(self, schema: DataSchema):
        self.schema = schema

    def validate(self, record: DataRecord) -> Tuple[bool, List[str]]:
        if isinstance(record.data, dict):
            return self.schema.validate(record.data)
        return True, []

    def validate_batch(self, records: List[DataRecord]) -> Tuple[List[DataRecord], List[DataRecord]]:
        valid = []
        invalid = []
        for record in records:
            is_valid, _ = self.validate(record)
            if is_valid:
                valid.append(record)
            else:
                invalid.append(record)
        return valid, invalid


class DataWriter:
    def __init__(self, output_path: str, format: DataFormat):
        self.output_path = output_path
        self.format = format
        self.buffer: List[DataRecord] = []
        self.buffer_size = 1000
        self.count = 0

    def write(self, record: DataRecord):
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def write_batch(self, records: List[DataRecord]):
        for record in records:
            self.write(record)

    def flush(self):
        if not self.buffer:
            return

        if self.format == DataFormat.JSON:
            self._write_json()
        elif self.format == DataFormat.CSV:
            self._write_csv()
        elif self.format == DataFormat.NPY:
            self._write_npy()
        elif self.format == DataFormat.PARQUET:
            self._write_parquet()

        self.buffer.clear()

    def _write_json(self):
        mode = 'a' if os.path.exists(self.output_path) else 'w'
        with open(self.output_path, mode) as f:
            for record in self.buffer:
                json.dump(record.data, f)
                f.write('\n')

    def _write_csv(self):
        import csv
        if not self.buffer:
            return

        fieldnames = list(self.buffer[0].data.keys()) if isinstance(self.buffer[0].data, dict) else ['data']
        mode = 'a' if os.path.exists(self.output_path) else 'w'

        with open(self.output_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()
            for record in self.buffer:
                if isinstance(record.data, dict):
                    writer.writerow(record.data)
                else:
                    writer.writerow({'data': str(record.data)})

    def _write_npy(self):
        data = np.array([r.data for r in self.buffer if isinstance(r.data, (int, float, np.ndarray))])
        if os.path.exists(self.output_path):
            existing = np.load(self.output_path)
            data = np.concatenate([existing, data])
        np.save(self.output_path, data)

    def _write_parquet(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            if not self.buffer:
                return

            data = [r.data for r in self.buffer if isinstance(r.data, dict)]
            if data:
                table = pa.Table.from_pylist(data)
                if os.path.exists(self.output_path):
                    existing = pq.read_table(self.output_path)
                    table = pa.concat_tables([existing, table])
                pq.write_table(table, self.output_path)
        except ImportError:
            pass

    def close(self):
        self.flush()


class IngestionPipeline:
    def __init__(
        self,
        reader: BaseReader,
        transformer: Optional[DataTransformer] = None,
        validator: Optional[DataValidator] = None,
        writer: Optional[DataWriter] = None,
        batch_size: int = 1000,
        num_workers: int = 4
    ):
        self.reader = reader
        self.transformer = transformer
        self.validator = validator
        self.writer = writer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        self.running = False
        self.stats = {
            'records_read': 0,
            'records_transformed': 0,
            'records_validated': 0,
            'records_written': 0,
            'records_failed': 0
        }
        self.lock = threading.Lock()

    def run(self, source_path: str) -> Dict[str, Any]:
        self.running = True
        start_time = time.time()

        try:
            for batch in self.reader.stream(source_path, self.batch_size):
                if not self.running:
                    break

                future = self.executor.submit(self._process_batch, batch)
                result = future.result()

                with self.lock:
                    self.stats['records_read'] += result['read']
                    self.stats['records_transformed'] += result['transformed']
                    self.stats['records_validated'] += result['validated']
                    self.stats['records_written'] += result['written']
                    self.stats['records_failed'] += result['failed']

        finally:
            self.running = False
            if self.writer:
                self.writer.close()

        elapsed = time.time() - start_time
        return {
            **self.stats,
            'elapsed_time': elapsed,
            'throughput': self.stats['records_read'] / elapsed if elapsed > 0 else 0
        }

    def _process_batch(self, batch: List[DataRecord]) -> Dict[str, int]:
        read_count = len(batch)
        transformed = batch

        if self.transformer:
            transformed = self.transformer.transform_batch(batch)

        validated = transformed
        failed = 0

        if self.validator:
            validated, invalid = self.validator.validate_batch(transformed)
            failed = len(invalid)

        written = 0
        if self.writer and validated:
            self.writer.write_batch(validated)
            written = len(validated)

        return {
            'read': read_count,
            'transformed': len(transformed),
            'validated': len(validated),
            'written': written,
            'failed': failed
        }

    def stop(self):
        self.running = False
        self.executor.shutdown(wait=True)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return self.stats.copy()


class IngestionScheduler:
    def __init__(self):
        self.jobs: Dict[str, IngestionJob] = {}
        self.sources: Dict[str, DataSource] = {}
        self.pipelines: Dict[str, IngestionPipeline] = {}
        self.lock = threading.Lock()

    def add_source(self, source: DataSource):
        with self.lock:
            self.sources[source.id] = source

    def create_pipeline(self, source_id: str, pipeline: IngestionPipeline):
        with self.lock:
            self.pipelines[source_id] = pipeline

    def schedule_job(self, source_id: str) -> IngestionJob:
        job = IngestionJob(
            id=f"job_{source_id}_{int(time.time())}",
            source_id=source_id
        )
        with self.lock:
            self.jobs[job.id] = job
        return job

    def run_job(self, job_id: str) -> IngestionJob:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return job

            job.status = IngestionStatus.RUNNING
            job.started_at = time.time()
            source = self.sources.get(job.source_id)
            pipeline = self.pipelines.get(job.source_id)

            if not source or not pipeline:
                job.status = IngestionStatus.FAILED
                job.error_message = "Source or pipeline not found"
                return job

        try:
            result = pipeline.run(source.path)

            with self.lock:
                job.status = IngestionStatus.COMPLETED
                job.completed_at = time.time()
                job.records_processed = result.get('records_read', 0)
                job.records_failed = result.get('records_failed', 0)
                job.progress = 1.0
                job.metadata = result

        except Exception as e:
            with self.lock:
                job.status = IngestionStatus.FAILED
                job.completed_at = time.time()
                job.error_message = str(e)

        return job

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = IngestionStatus.CANCELLED
                return True
        return False


class IngestionMetrics:
    def __init__(self):
        self.total_records = 0
        self.total_bytes = 0
        self.total_time = 0.0
        self.errors = 0
        self.lock = threading.Lock()

    def record_ingestion(self, records: int, bytes_count: int, elapsed: float):
        with self.lock:
            self.total_records += records
            self.total_bytes += bytes_count
            self.total_time += elapsed

    def record_error(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'total_records': self.total_records,
                'total_bytes': self.total_bytes,
                'total_time': self.total_time,
                'avg_throughput': self.total_records / max(self.total_time, 1e-6),
                'errors': self.errors
            }


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet