#!/usr/bin/env python3
"""Focused regression checks for launcher terminal and topology behavior."""

import errno
import importlib.util
import os
import pty
import select
import signal
import subprocess
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "recommend_gpu_topology", ROOT / "tools/recommend_gpu_topology.py"
)
topology = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topology)


def run_tty(command, exchanges, timeout=10):
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.environ["TERM"] = "xterm"
        os.execv("/bin/bash", ["bash", "-c", command])
    output = b""
    search_start = 0
    status = None
    try:
        for marker, keys in exchanges:
            deadline = time.monotonic() + timeout
            while marker.encode() not in output[search_start:]:
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"PTY prompt not seen: {marker}; output={output[-800:]!r}"
                    )
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        output += os.read(master, 65536)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        raise AssertionError(
                            f"PTY child exited before prompt {marker}: "
                            f"{output[-800:]!r}"
                        ) from exc
            for key in keys:
                try:
                    os.write(master, key)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    raise AssertionError(
                        f"PTY child exited before key input: {output[-800:]!r}"
                    ) from exc
                time.sleep(0.04)
            search_start = len(output)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waited, child_status = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                status = child_status
                break
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    output += os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
        if status is None:
            raise AssertionError(f"PTY child did not exit: {output[-800:]!r}")
        while select.select([master], [], [], 0)[0]:
            try:
                output += os.read(master, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
        return output.decode(errors="replace"), os.waitstatus_to_exitcode(status)
    finally:
        if status is None:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                os.waitpid(pid, 0)
        os.close(master)


class LauncherInteractiveTest(unittest.TestCase):
    def test_tty_runner_times_out_and_reaps_child(self):
        with self.assertRaisesRegex(AssertionError, "did not exit"):
            run_tty("sleep 30", [], timeout=0.2)

    def test_main_menu_wraps_with_split_escape_bytes(self):
        for start, arrow, expected in (
            (1, b"A", 0),
            (0, b"A", 9),
            (9, b"B", 0),
            (0, b"B", 1),
        ):
            with self.subTest(start=start, arrow=arrow):
                output, code = run_tty(
                    f"source ./launcher.sh; choice=$(read_main_menu_choice {start}); "
                    'printf "RESULT:%s\\n" "$choice"',
                    [("", [b"\x1b", b"[", arrow])],
                )
                self.assertEqual(code, 0, output)
                self.assertIn(f"RESULT:__INDEX__:{expected}", output)

    def test_startup_escape_requires_confirmation(self):
        with tempfile.NamedTemporaryFile() as log:
            output, code = run_tty(
                f"source ./launcher.sh; set +e; START_TIMEOUT=20; PORT=0; "
                f"CURRENT_SERVER_PID=$$; wait_for_ready {log.name} 127.0.0.1; "
                'printf "RESULT:%s\\n" "$?"',
                [
                    ("Press Esc to cancel startup.", [b"\x1b"]),
                    ("Terminate startup and stop the new service?", [b"y", b"\r"]),
                ],
            )
            self.assertEqual(code, 0, output)
            self.assertIn("RESULT:130", output)

    def test_startup_escape_declined_keeps_waiting(self):
        with tempfile.NamedTemporaryFile() as log:
            output, code = run_tty(
                f"source ./launcher.sh; set +e; START_TIMEOUT=20; PORT=0; "
                f"CURRENT_SERVER_PID=$$; wait_for_ready {log.name} 127.0.0.1; "
                'printf "RESULT:%s\\n" "$?"',
                [
                    ("Press Esc to cancel startup.", [b"\x1b"]),
                    ("Terminate startup and stop the new service?", [b"n", b"\r"]),
                    ("Press Esc to cancel startup.", [b"\x1b"]),
                    ("Terminate startup and stop the new service?", [b"y", b"\r"]),
                ],
            )
            self.assertEqual(code, 0, output)
            self.assertIn("RESULT:130", output)

    def test_model_history_is_bounded_and_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / str(index) for index in range(12)]
            for path in paths:
                path.mkdir()
            script = "source ./launcher.sh; " + " ".join(
                f"record_model_history target {path};" for path in paths
            )
            script += f" record_model_history target {paths[4]};"
            script += f" record_model_history draft {paths[0]}"
            subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={**os.environ, "LOG_DIR": directory},
                check=True,
            )
            target = (Path(directory) / "model-target-history").read_text().splitlines()
            draft = (Path(directory) / "model-draft-history").read_text().splitlines()
            self.assertEqual(len(target), 10)
            self.assertEqual(target[0], str(paths[4]))
            self.assertEqual(len(set(target)), 10)
            self.assertEqual(draft, [str(paths[0])])
            history = Path(directory) / "model-target-history"
            history.write_text(
                f"{paths[4]}\n{paths[4]}\n{directory}/missing\n{paths[5]}\n"
            )
            subprocess.run(
                ["bash", "-c", 'source ./launcher.sh; record_model_history target ""'],
                cwd=ROOT,
                env={**os.environ, "LOG_DIR": directory},
                check=True,
            )
            self.assertEqual(
                history.read_text().splitlines(), [str(paths[4]), str(paths[5])]
            )

    def test_cancel_cleanup_only_stops_recorded_process(self):
        with tempfile.TemporaryDirectory() as directory:
            startup = subprocess.Popen(["sleep", "30"], start_new_session=True)
            unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
            pid_file = Path(directory) / "startup.pid"
            pid_file.write_text(f"{startup.pid}\n")
            try:
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source ./launcher.sh; cleanup_cancelled_launch "$1"',
                        "_",
                        str(pid_file),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                startup.wait(timeout=5)
                self.assertIsNone(unrelated.poll())
                self.assertFalse(pid_file.exists())
            finally:
                for process in (startup, unrelated):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)

    def test_cancel_cleanup_finds_reparented_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            token = "launcher-test-orphan-worker"
            parent = subprocess.run(
                ["bash", "-c", 'sleep 30 >/dev/null 2>&1 & printf "%s\\n" "$!"'],
                capture_output=True,
                text=True,
                env={**os.environ, "VLLM_LAUNCH_TOKEN": token},
                start_new_session=True,
                check=True,
            )
            worker_pid = int(parent.stdout.strip())
            unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
            pid_file = Path(directory) / "startup.pid"
            pid_file.write_text("999999\n")
            Path(f"{pid_file}.token").write_text(f"{token}\n")
            try:
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source ./launcher.sh; cleanup_cancelled_launch "$1"',
                        "_",
                        str(pid_file),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertIsNone(unrelated.poll())
                self.assertFalse(pid_file.exists())
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(worker_pid)],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.assertTrue(not state or state.startswith("Z"), state)
            finally:
                if unrelated.poll() is None:
                    unrelated.terminate()
                unrelated.wait(timeout=5)
                with suppress(ProcessLookupError):
                    os.kill(worker_pid, 9)

    def test_failed_cancel_keeps_record(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "startup.pid"
            pid_file.write_text("999999\n")
            Path(f"{pid_file}.token").write_text("test-token\n")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "source ./launcher.sh; "
                    "launch_token_pids() { echo 999998; }; "
                    'cleanup_cancelled_launch "$1"',
                    "_",
                    str(pid_file),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(pid_file.exists())
            self.assertTrue(Path(f"{pid_file}.token").exists())

    def test_noninteractive_tty_does_not_read_startup_keys(self):
        output, code = run_tty(
            "source ./launcher.sh; NON_INTERACTIVE=1; "
            "if startup_can_read_tty; then echo RESULT:yes; else echo RESULT:no; fi",
            [],
        )
        self.assertEqual(code, 0, output)
        self.assertIn("RESULT:no", output)

    def test_startup_benchmark_prefers_bundled_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory) / "tools"
            tools.mkdir()
            (tools / "profile_request.py").touch()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "source ./launcher.sh; startup_performance_helper_path",
                ],
                cwd=ROOT,
                env={**os.environ, "RUNTIME_ROOT": directory},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                Path(result.stdout.strip()), ROOT / "tools" / "profile_request.py"
            )

    def test_topology_groups_preserve_p2p(self):
        matrix = "\x1b[4mGPU0 GPU1 GPU2 GPU3 CPU Affinity\x1b[0m\n"
        matrix += "GPU0 X PHB PIX PHB 0-3\nGPU1 PHB X PHB PIX 0-3\n"
        matrix += "GPU2 PIX PHB X PHB 0-3\nGPU3 PHB PIX PHB X 0-3\n"
        links = topology.read_matrix(matrix)
        p2p = {
            a: {
                b: "OK"
                if a == b or {a, b} in ({"GPU0", "GPU2"}, {"GPU1", "GPU3"})
                else "CNS"
                for b in links
            }
            for a in links
        }
        result = topology.recommend(["0", "1", "2", "3"], 2, links, p2p)
        self.assertEqual(result["ordered_devices"], "0,2,1,3")

    def test_tp1_pipeline_stages_follow_fast_links(self):
        names = [f"GPU{index}" for index in range(4)]
        links = {
            left: {right: "X" if left == right else "SYS" for right in names}
            for left in names
        }
        for left, right in (("GPU0", "GPU2"), ("GPU2", "GPU1"), ("GPU1", "GPU3")):
            links[left][right] = links[right][left] = "NV2"
        result = topology.recommend(["0", "1", "2", "3"], 1, links, pp_size=4)
        self.assertEqual(result["ordered_devices"], "0,2,1,3")

    def test_equal_pipeline_links_keep_selected_order(self):
        names = [f"GPU{index}" for index in range(4)]
        links = {
            left: {right: "X" if left == right else "PIX" for right in names}
            for left in names
        }
        result = topology.recommend(["3", "1", "0", "2"], 1, links, pp_size=4)
        self.assertEqual(result["ordered_devices"], "3,1,0,2")

    def test_rank_layout_labels_pipeline_stages(self):
        result = subprocess.run(
            ["bash", "-c", 'source ./launcher.sh; format_tp_rank_groups "1,5,6,7" 2'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "PP0(TP2)=[1,5]  PP1(TP2)=[6,7]")

    def test_pipeline_stages_align_corresponding_tp_ranks(self):
        names = [f"GPU{index}" for index in range(4)]
        links = {
            left: {right: "X" if left == right else "PHB" for right in names}
            for left in names
        }
        p2p = {
            left: {right: "OK" if left == right else "CNS" for right in names}
            for left in names
        }
        for left, right in (("GPU0", "GPU1"), ("GPU2", "GPU3")):
            links[left][right] = links[right][left] = "PIX"
            p2p[left][right] = p2p[right][left] = "OK"
        for left, right in (("GPU0", "GPU3"), ("GPU1", "GPU2")):
            links[left][right] = links[right][left] = "NV2"
        result = topology.recommend(["0", "1", "2", "3"], 2, links, p2p, 2)
        self.assertEqual(result["ordered_devices"], "0,1,3,2")

    def test_pipeline_stage_order_can_change_after_tp_grouping(self):
        names = [f"GPU{index}" for index in range(6)]
        links = {
            left: {right: "X" if left == right else "SYS" for right in names}
            for left in names
        }
        p2p = {
            left: {right: "OK" if left == right else "CNS" for right in names}
            for left in names
        }
        for left, right in (("GPU0", "GPU1"), ("GPU2", "GPU3"), ("GPU4", "GPU5")):
            links[left][right] = links[right][left] = "PIX"
            p2p[left][right] = p2p[right][left] = "OK"
        for left, right in (
            ("GPU0", "GPU4"),
            ("GPU1", "GPU5"),
            ("GPU4", "GPU2"),
            ("GPU5", "GPU3"),
        ):
            links[left][right] = links[right][left] = "NV2"
        result = topology.recommend(
            [str(index) for index in range(6)], 2, links, p2p, 3
        )
        self.assertEqual(result["ordered_devices"], "0,1,4,5,2,3")

    def test_topology_search_has_device_bound(self):
        with self.assertRaisesRegex(ValueError, "at most 12"):
            topology.recommend([str(index) for index in range(13)], 1, {})

    def test_single_gpu_topology_header(self):
        matrix = "\x1b[4mGPU0 CPU Affinity\x1b[0m\nGPU0 X 0-3\n"
        result = topology.recommend(["0"], 1, topology.read_matrix(matrix))
        self.assertEqual(result["ordered_devices"], "0")


if __name__ == "__main__":
    unittest.main()
