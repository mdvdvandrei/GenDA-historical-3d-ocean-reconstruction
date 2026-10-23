# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import json


SAVE_NETWORK_SNAPSHOT = "save network snapshot"
SAVE_TRAINING_STATE = "save training state"
WANDB_ID = "wandb id"


@dataclasses.dataclass
class EventLog:
    file: str

    def _log_obj(self, obj):
        with open(self.file, "a") as f:
            print(json.dumps(obj), file=f)

    def log_training_state(self, filename: str, nimg: int):
        self._log_obj(
            {"event": SAVE_TRAINING_STATE, "filename": filename, "nimg": nimg}
        )

    def log_network_snapshot(self, filename: str, nimg: int):
        self._log_obj(
            {"event": SAVE_NETWORK_SNAPSHOT, "filename": filename, "nimg": nimg}
        )

    def log_wandb_id(self, id: str):
        return self._log_obj({"event": WANDB_ID, "id": id})

    def _iter_events(self):
        with open(self.file, "r") as f:
            for line in f:
                event = json.loads(line)
                yield event

    def query(self, event_type):
        for event in self._iter_events():
            if event["event"] == event_type:
                yield event

    def last_state(self):
        found = None
        with open(self.file, "r") as f:
            for line in f:
                event = json.loads(line)
                if event["event"] == SAVE_TRAINING_STATE:
                    found = event["filename"], event["nimg"]
        return found

    def states(self):
        for event in self._iter_events():
            if event["event"] == SAVE_TRAINING_STATE:
                yield event["filename"], event["nimg"]

    def get_wandb_id(self):
        for event in self._iter_events():
            if event["event"] == WANDB_ID:
                return event["id"]
