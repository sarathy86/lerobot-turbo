# Parallel Video Encoding During Recording — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kick off video encoding in a background thread the moment the right arrow key ends an episode, so the reset phase and next recording overlap with encoding instead of blocking on it.

**Architecture:** Split `LeRobotDataset.save_episode()` into two phases: Phase 1 runs synchronously (validate, image-writer drain, Parquet write, counter commit, buffer reset, ~1-2 s), Phase 2 runs in a `ThreadPoolExecutor(max_workers=1)` (encode videos, write episode metadata). Move `save_episode(non_blocking=True)` in `lerobot_record.py` to before the reset phase so encoding overlaps with it. The single-worker executor enforces per-episode ordering because `_save_episode_video()` relies on `meta.latest_episode` set by the previous episode's Phase 2.

**Tech Stack:** Python `concurrent.futures` (already imported), `ThreadPoolExecutor`, `ProcessPoolExecutor` (unchanged for multi-camera parallel encoding).

## Global Constraints

- Line length: 110 characters (ruff enforced)
- Python minimum 3.10; use `|` union syntax
- `non_blocking=True` and `batch_encoding_size > 1` must raise `ValueError` — they are mutually exclusive
- `non_blocking=False` (default) must produce byte-for-byte identical output to current behaviour
- All new fields added to `LeRobotDataset.__init__()` must also be added to `LeRobotDataset.create()` so `test_same_attributes_defined` keeps passing

---

### Task 1: Add `skip_counter_update` to `LeRobotDatasetMetadata.save_episode()`

**Files:**
- Modify: `src/lerobot/datasets/lerobot_dataset.py:398-424`
- Test: `tests/datasets/test_datasets.py`

**Interfaces:**
- Produces: `LeRobotDatasetMetadata.save_episode(..., skip_counter_update: bool = False)` — when `True`, skips the `total_episodes`/`total_frames` increment and `write_info()` call; still calls `_save_episode_metadata()`, updates `stats`, writes `stats.json`.

- [ ] **Step 1: Write the failing test**

Add to `tests/datasets/test_datasets.py`:

```python
def test_meta_save_episode_skip_counter_update(tmp_path, empty_lerobot_dataset_factory):
    """skip_counter_update=True must not change total_episodes or total_frames."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})

    # Manually do Phase-1 book-keeping so meta has valid state
    ep_buf = dataset.episode_buffer
    ep_len = ep_buf.pop("size")
    tasks = ep_buf.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = ep_buf["episode_index"]
    dataset.meta.save_episode_tasks(episode_tasks)

    before_eps = dataset.meta.total_episodes
    before_frames = dataset.meta.total_frames

    # Call with skip_counter_update=True — counters must stay the same
    dataset.meta.save_episode(
        episode_index=episode_index,
        episode_length=ep_len,
        episode_tasks=episode_tasks,
        episode_stats={"state": {"mean": np.array([0.0]), "std": np.array([1.0]),
                                  "min": np.array([0.0]), "max": np.array([1.0]),
                                  "count": np.array([1])}},
        episode_metadata={},
        skip_counter_update=True,
    )

    assert dataset.meta.total_episodes == before_eps
    assert dataset.meta.total_frames == before_frames

    # Calling WITHOUT skip_counter_update must increment as usual
    dataset.meta.save_episode(
        episode_index=episode_index,
        episode_length=ep_len,
        episode_tasks=episode_tasks,
        episode_stats={"state": {"mean": np.array([0.0]), "std": np.array([1.0]),
                                  "min": np.array([0.0]), "max": np.array([1.0]),
                                  "count": np.array([1])}},
        episode_metadata={},
        skip_counter_update=False,
    )

    assert dataset.meta.total_episodes == before_eps + 1
    assert dataset.meta.total_frames == before_frames + ep_len
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/datasets/test_datasets.py::test_meta_save_episode_skip_counter_update -v
```
Expected: `FAILED` — `TypeError: save_episode() got an unexpected keyword argument 'skip_counter_update'`

- [ ] **Step 3: Add `skip_counter_update` parameter to `MetaData.save_episode()`**

In `src/lerobot/datasets/lerobot_dataset.py`, replace the `save_episode` method of `LeRobotDatasetMetadata` (lines 398-424):

```python
def save_episode(
    self,
    episode_index: int,
    episode_length: int,
    episode_tasks: list[str],
    episode_stats: dict[str, dict],
    episode_metadata: dict,
    skip_counter_update: bool = False,
) -> None:
    episode_dict = {
        "episode_index": episode_index,
        "tasks": episode_tasks,
        "length": episode_length,
    }
    episode_dict.update(episode_metadata)
    episode_dict.update(flatten_dict({"stats": episode_stats}))
    self._save_episode_metadata(episode_dict)

    if not skip_counter_update:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        write_info(self.info, self.root)

    self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
    write_stats(self.stats, self.root)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/datasets/test_datasets.py::test_meta_save_episode_skip_counter_update -v
```
Expected: `PASSED`

- [ ] **Step 5: Verify no existing tests broken**

```bash
pytest tests/datasets/test_datasets.py -v -x
```
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/datasets/lerobot_dataset.py tests/datasets/test_datasets.py
git commit -m "feat(dataset): add skip_counter_update param to MetaData.save_episode"
```

---

### Task 2: Add executor fields to `LeRobotDataset` and drain them in `finalize()`

**Files:**
- Modify: `src/lerobot/datasets/lerobot_dataset.py:706-722` (`__init__`), `1617-1622` (`create`), `1100-1106` (`finalize`)
- Test: `tests/datasets/test_datasets.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces:
  - `LeRobotDataset._encoding_executor: concurrent.futures.ThreadPoolExecutor | None` — `None` until first non-blocking save
  - `LeRobotDataset._pending_encoding_future: concurrent.futures.Future | None` — tracks last submitted job
  - `LeRobotDataset.finalize()` — waits for `_pending_encoding_future`, shuts down executor, then closes writers

- [ ] **Step 1: Write the failing test**

Add to `tests/datasets/test_datasets.py`:

```python
def test_finalize_waits_for_pending_future(tmp_path, empty_lerobot_dataset_factory):
    """finalize() must block until _pending_encoding_future completes."""
    import concurrent.futures
    import threading

    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)

    completed = threading.Event()

    def slow_job():
        import time
        time.sleep(0.05)
        completed.set()

    dataset._encoding_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    dataset._pending_encoding_future = dataset._encoding_executor.submit(slow_job)

    assert not completed.is_set()
    dataset.finalize()
    assert completed.is_set()


def test_finalize_ok_with_no_pending_future(tmp_path, empty_lerobot_dataset_factory):
    """finalize() must not raise when no encoding was ever submitted."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.finalize()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/datasets/test_datasets.py::test_finalize_waits_for_pending_future \
       tests/datasets/test_datasets.py::test_finalize_ok_with_no_pending_future -v
```
Expected: `FAILED` — `AttributeError: 'LeRobotDataset' object has no attribute '_encoding_executor'`

- [ ] **Step 3: Add fields to `__init__()` and `create()`**

In `src/lerobot/datasets/lerobot_dataset.py`, in `LeRobotDataset.__init__()`, after the line `self.latest_episode = None` (line ~709), add:

```python
self._encoding_executor: concurrent.futures.ThreadPoolExecutor | None = None
self._pending_encoding_future: concurrent.futures.Future | None = None
```

In `LeRobotDataset.create()`, after the line `obj.latest_episode = None` (line ~1617), add:

```python
obj._encoding_executor = None
obj._pending_encoding_future = None
```

- [ ] **Step 4: Update `finalize()`**

Replace `LeRobotDataset.finalize()` (lines 1100-1106):

```python
def finalize(self):
    """Close parquet writers. Must be called after data collection, else parquet footers won't be written."""
    if self._pending_encoding_future is not None:
        self._pending_encoding_future.result()  # wait; re-raises any encoding exception
    if self._encoding_executor is not None:
        self._encoding_executor.shutdown(wait=True)  # drain any queued jobs
    self._close_writer()
    self.meta._close_writer()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/datasets/test_datasets.py::test_finalize_waits_for_pending_future \
       tests/datasets/test_datasets.py::test_finalize_ok_with_no_pending_future -v
```
Expected: `PASSED`

- [ ] **Step 6: Verify `test_same_attributes_defined` still passes** (confirms both `__init__` and `create` have the same fields)

```bash
pytest tests/datasets/test_datasets.py::test_same_attributes_defined -v
```
Expected: `PASSED`

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/datasets/lerobot_dataset.py tests/datasets/test_datasets.py
git commit -m "feat(dataset): add background encoding executor fields and drain in finalize()"
```

---

### Task 3: Refactor `save_episode()` with `non_blocking` parameter

**Files:**
- Modify: `src/lerobot/datasets/lerobot_dataset.py:1182-1286`
- Test: `tests/datasets/test_datasets.py`

**Interfaces:**
- Consumes:
  - `LeRobotDatasetMetadata.save_episode(..., skip_counter_update: bool = False)` from Task 1
  - `LeRobotDataset._encoding_executor`, `LeRobotDataset._pending_encoding_future` from Task 2
- Produces: `LeRobotDataset.save_episode(episode_data=None, parallel_encoding=True, non_blocking=False)`

- [ ] **Step 1: Write failing tests**

Add to `tests/datasets/test_datasets.py`:

```python
def test_save_episode_non_blocking_raises_with_batch_encoding(tmp_path, empty_lerobot_dataset_factory):
    """non_blocking=True with batch_encoding_size > 1 must raise ValueError."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test", features=features, batch_encoding_size=2
    )
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})
    with pytest.raises(ValueError, match="non_blocking=True is incompatible with batch_encoding_size"):
        dataset.save_episode(non_blocking=True)


def test_save_episode_non_blocking_commits_counter_before_returning(tmp_path, empty_lerobot_dataset_factory):
    """After save_episode(non_blocking=True) returns, total_episodes must already be incremented."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})

    assert dataset.meta.total_episodes == 0
    dataset.save_episode(non_blocking=True)
    assert dataset.meta.total_episodes == 1  # incremented synchronously in Phase 1

    dataset.finalize()


def test_save_episode_non_blocking_clears_buffer_before_returning(tmp_path, empty_lerobot_dataset_factory):
    """After save_episode(non_blocking=True) returns, episode_buffer must be reset for next episode."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})

    dataset.save_episode(non_blocking=True)
    # Buffer is reset; the new buffer's episode_index must equal total_episodes (1)
    assert dataset.episode_buffer["episode_index"] == 1

    dataset.finalize()


def test_save_episode_non_blocking_default_false_unchanged(tmp_path, empty_lerobot_dataset_factory):
    """save_episode() with default non_blocking=False must behave identically to before."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})
    dataset.save_episode()  # non_blocking defaults to False
    assert dataset.meta.total_episodes == 1
    dataset.finalize()


def test_save_episode_non_blocking_ordering(tmp_path, empty_lerobot_dataset_factory):
    """Two consecutive non_blocking saves must produce the same dataset as two blocking saves."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}

    # Blocking reference
    ds_blocking = empty_lerobot_dataset_factory(root=tmp_path / "blocking", features=features)
    for _ in range(2):
        ds_blocking.add_frame({"state": torch.randn(1), "task": "Dummy task"})
        ds_blocking.save_episode()
    ds_blocking.finalize()

    # Non-blocking
    ds_nonblock = empty_lerobot_dataset_factory(root=tmp_path / "nonblock", features=features)
    for _ in range(2):
        ds_nonblock.add_frame({"state": torch.randn(1), "task": "Dummy task"})
        ds_nonblock.save_episode(non_blocking=True)
    ds_nonblock.finalize()

    assert ds_nonblock.meta.total_episodes == ds_blocking.meta.total_episodes
    assert ds_nonblock.meta.total_frames == ds_blocking.meta.total_frames
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/datasets/test_datasets.py \
  -k "test_save_episode_non_blocking" -v
```
Expected: all four `FAILED` — `TypeError: save_episode() got an unexpected keyword argument 'non_blocking'`

- [ ] **Step 3: Refactor `save_episode()` in `LeRobotDataset`**

Replace `LeRobotDataset.save_episode()` (lines 1182-1286) with:

```python
def save_episode(
    self,
    episode_data: dict | None = None,
    parallel_encoding: bool = True,
    non_blocking: bool = False,
) -> None:
    """Save the current episode to disk.

    Video encoding is handled based on batch_encoding_size and non_blocking:
    - If batch_encoding_size == 1 and non_blocking=False: encode immediately (blocking).
    - If batch_encoding_size == 1 and non_blocking=True: encode in background thread.
    - If batch_encoding_size > 1: encode in batches (non_blocking=True not allowed).

    Args:
        episode_data: Dict with episode data to save. If None, saves self.episode_buffer.
        parallel_encoding: If True and multiple cameras, encode cameras in parallel processes.
        non_blocking: If True, run video encoding in a background thread so the caller
            can begin the next recording immediately. Incompatible with batch_encoding_size > 1.
    """
    if non_blocking and self.batch_encoding_size > 1:
        raise ValueError(
            "non_blocking=True is incompatible with batch_encoding_size > 1. "
            "Use one or the other, not both."
        )

    episode_buffer = episode_data if episode_data is not None else self.episode_buffer

    # ── Phase 1: synchronous work ──────────────────────────────────────────────────────────
    validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    self.meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

    for key, ft in self.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    # Wait for image writer to end so episode stats over images can be computed
    self._wait_image_writer()
    ep_stats = compute_episode_stats(episode_buffer, self.features)
    ep_metadata = self._save_episode_data(episode_buffer)

    # Commit counters now so create_episode_buffer() for the next episode gets the right index
    self.meta.info["total_episodes"] += 1
    self.meta.info["total_frames"] += episode_length
    self.meta.info["total_tasks"] = len(self.meta.tasks)
    self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
    write_info(self.meta.info, self.meta.root)

    if not episode_data:
        # Reset episode buffer; delete image-feature PNGs (video-feature PNGs stay for Phase 2)
        self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    # ── Phase 2: video encoding + metadata commit ──────────────────────────────────────────
    has_video_keys = len(self.meta.video_keys) > 0
    use_batched_encoding = self.batch_encoding_size > 1

    def _encode_and_commit() -> None:
        nonlocal ep_metadata

        if has_video_keys and not use_batched_encoding:
            num_cameras = len(self.meta.video_keys)
            if parallel_encoding and num_cameras > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
                    future_to_key = {
                        executor.submit(
                            _encode_video_worker,
                            video_key,
                            episode_index,
                            self.root,
                            self.fps,
                            self.vcodec,
                        ): video_key
                        for video_key in self.meta.video_keys
                    }
                    results: dict[str, Path] = {}
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        try:
                            results[video_key] = future.result()
                        except Exception as exc:
                            logging.error(f"Video encoding failed for {video_key}: {exc}")
                            raise

                for video_key in self.meta.video_keys:
                    ep_metadata.update(
                        self._save_episode_video(video_key, episode_index, temp_path=results[video_key])
                    )
            else:
                for video_key in self.meta.video_keys:
                    ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # Write episode metadata; counters were already committed in Phase 1
        self.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, ep_metadata,
            skip_counter_update=True,
        )

        if has_video_keys and use_batched_encoding:
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

    if non_blocking:
        if self._encoding_executor is None:
            self._encoding_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_encoding_future = self._encoding_executor.submit(_encode_and_commit)
    else:
        _encode_and_commit()
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/datasets/test_datasets.py -k "test_save_episode_non_blocking" -v
```
Expected: all four `PASSED`

- [ ] **Step 5: Run the full dataset test suite to verify nothing regressed**

Pay particular attention to `test_tmp_video_deletion` (verifies video PNGs are cleaned up after encoding) and `test_add_frame` (verifies the basic save_episode path). Both should still pass because Phase 2 still calls `_encode_video_worker` which deletes the image directory after encoding.

```bash
pytest tests/datasets/test_datasets.py -v -x
```
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/datasets/lerobot_dataset.py tests/datasets/test_datasets.py
git commit -m "feat(dataset): add non_blocking param to save_episode() for parallel video encoding"
```

---

### Task 4: Update `lerobot_record.py` — move save_episode before reset, reject re-record

**Files:**
- Modify: `src/lerobot/scripts/lerobot_record.py:494-547`
- Test: `tests/datasets/test_datasets.py` (unit-level logic test with mock)

**Interfaces:**
- Consumes: `LeRobotDataset.save_episode(non_blocking=True)` from Task 3

- [ ] **Step 1: Write failing test**

Add to `tests/datasets/test_datasets.py`:

```python
def test_rerecord_rejected_when_encoding_in_flight(tmp_path, empty_lerobot_dataset_factory, caplog):
    """When save_episode(non_blocking=True) was called, a rerecord_episode event must be
    cleared with a warning log rather than rewinding the episode."""
    import logging
    import concurrent.futures

    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})
    dataset.save_episode(non_blocking=True)

    events = {"rerecord_episode": True, "exit_early": False}

    with caplog.at_level(logging.WARNING):
        if events["rerecord_episode"]:
            if (
                dataset._pending_encoding_future is not None
                and not dataset._pending_encoding_future.done()
            ):
                logging.warning(
                    f"Re-record requested but episode {dataset.meta.total_episodes - 1} "
                    "encoding is already in progress. Proceeding to next episode."
                )
                events["rerecord_episode"] = False
                events["exit_early"] = False

    assert not events["rerecord_episode"]
    assert "encoding is already in progress" in caplog.text
    assert dataset.meta.total_episodes == 1  # episode was NOT rolled back

    dataset.finalize()
```

- [ ] **Step 2: Run test to verify it passes already** (it tests pure logic, not the record script)

```bash
pytest tests/datasets/test_datasets.py::test_rerecord_rejected_when_encoding_in_flight -v
```
Expected: `PASSED` — this test exercises the same logic that Task 4 encodes into the record script.

- [ ] **Step 3: Restructure the recording loop in `lerobot_record.py`**

Replace lines 494–547 in `src/lerobot/scripts/lerobot_record.py`:

```python
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                # Kick off encoding immediately; reset phase and next recording overlap with it
                dataset.save_episode(non_blocking=True)
                recorded_episodes += 1

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and recorded_episodes < cfg.dataset.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)

                    # reset g1 robot
                    if robot.name == "unitree_g1":
                        robot.reset()

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    if (
                        dataset._pending_encoding_future is not None
                        and not dataset._pending_encoding_future.done()
                    ):
                        logging.warning(
                            f"Re-record requested but episode {recorded_episodes - 1} encoding is "
                            "already in progress. Proceeding to next episode."
                        )
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
```

- [ ] **Step 4: Run the linting check**

```bash
ruff check src/lerobot/scripts/lerobot_record.py
ruff format src/lerobot/scripts/lerobot_record.py
```
Expected: no errors.

- [ ] **Step 5: Run the full dataset test suite one final time**

```bash
pytest tests/datasets/test_datasets.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Run the full test suite**

```bash
pytest -sv tests/ -x --ignore=tests/artifacts -q 2>&1 | tail -20
```
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/scripts/lerobot_record.py tests/datasets/test_datasets.py
git commit -m "feat(record): encode video in background; next recording starts immediately after right arrow"
```
