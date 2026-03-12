from __future__ import annotations

from collections import defaultdict
import csv
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from citylearn.agents.base import Agent
    from citylearn.citylearn import CityLearnEnv
    from citylearn.electric_vehicle import ElectricVehicle

LOGGER = logging.getLogger(__name__)


class EpisodeExporter:
    """Internal helper that owns rendering/export behaviour for ``CityLearnEnv``."""

    DEFAULT_RENDER_START_DATE = datetime.date(2024, 1, 1)

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    @staticmethod
    def parse_render_start_date(start_date: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
        """Return a valid start date for rendering timestamps."""

        if start_date is None:
            return EpisodeExporter.DEFAULT_RENDER_START_DATE

        if isinstance(start_date, datetime.datetime):
            return start_date.date()

        if isinstance(start_date, datetime.date):
            return start_date

        if isinstance(start_date, str):
            try:
                return datetime.date.fromisoformat(start_date)
            except ValueError as exc:
                raise ValueError(
                    "CityLearnEnv start_date must be in ISO format 'YYYY-MM-DD'."
                ) from exc

        raise TypeError(
            "CityLearnEnv start_date must be a date, datetime, or ISO format string."
        )

    def export_final_kpis(self, model: "Agent" = None, filepath: str = "exported_kpis.csv"):
        """Export episode KPIs to csv."""

        env = self.env
        self.ensure_output_dir()
        file_path = os.path.join(env.new_folder_path, filepath)

        if model is not None and getattr(model, 'env', None) is not None:
            kpis = model.env.evaluate()
        else:
            kpis = env.evaluate()

        kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
        kpis = kpis.dropna(how='all')
        kpis = kpis.fillna('')
        kpis = kpis.reset_index()
        kpis = kpis.rename(columns={'cost_function': 'KPI'})
        kpis.to_csv(file_path, index=False, encoding='utf-8')
        env._final_kpis_exported = True

    def render(self):
        """Render one time step to CSV outputs."""

        env = self.env

        if not getattr(env, 'render_enabled', False):
            return

        if env.render_mode == 'end' and getattr(env, '_defer_render_flush', False):
            return

        if env.render_mode == 'end' and (env.terminated or env.truncated):
            return

        self.ensure_output_dir()
        iso_timestamp = self.get_iso_timestamp()
        os.makedirs(env.new_folder_path, exist_ok=True)

        episode_num = env.episode_tracker.episode

        self.save_to_csv(
            f"exported_data_community_ep{episode_num}.csv",
            {"timestamp": iso_timestamp, **env.as_dict()},
        )

        for building in env.buildings:
            self.save_to_csv(
                f"exported_data_{building.name.lower()}_ep{episode_num}.csv",
                {"timestamp": iso_timestamp, **building.as_dict()},
            )

            battery = building.electrical_storage
            self.save_to_csv(
                f"exported_data_{building.name.lower()}_battery_ep{episode_num}.csv",
                {"timestamp": iso_timestamp, **battery.as_dict()},
            )

            for charger in building.electric_vehicle_chargers or []:
                self.save_to_csv(
                    f"exported_data_{building.name.lower()}_{charger.charger_id}_ep{episode_num}.csv",
                    {"timestamp": iso_timestamp, **charger.as_dict()},
                )

        self.save_to_csv(
            f"exported_data_pricing_ep{episode_num}.csv",
            {"timestamp": iso_timestamp, **env.buildings[0].pricing.as_dict(env.time_step)},
        )

        for ev in env.electric_vehicles:
            self.save_to_csv(
                f"exported_data_{ev.name.lower()}_ep{episode_num}.csv",
                {"timestamp": iso_timestamp, **ev.as_dict()},
            )

    def _set_charger_render_state(self, charger, time_step: int, ev_lookup: Mapping[str, "ElectricVehicle"]):
        """Set charger connected/incoming EV pointers to match schedule at a given time step."""

        sim = charger.charger_simulation
        state = sim.electric_vehicle_charger_state[time_step] if time_step < len(sim.electric_vehicle_charger_state) else np.nan
        ev_id = sim.electric_vehicle_id[time_step] if time_step < len(sim.electric_vehicle_id) else None
        valid_ev_id = isinstance(ev_id, str) and ev_id.strip() not in {"", "nan"}

        connected_ev = ev_lookup.get(ev_id) if valid_ev_id and state == 1 else None
        incoming_ev = ev_lookup.get(ev_id) if valid_ev_id and state == 2 else None

        charger.connected_electric_vehicle = connected_ev
        charger.incoming_electric_vehicle = incoming_ev

    def export_episode_render_data(self, final_index: int):
        """Export full episode render rows in one pass for ``render_mode='end'``."""

        env = self.env

        if final_index < 0:
            return

        self.ensure_output_dir()
        episode_num = env.episode_tracker.episode
        rows_by_filename: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        ev_lookup = {ev.name: ev for ev in env.electric_vehicles}
        original_charger_state = {}
        time_step_snapshot = self.override_render_time_step(0)
        original_year = env.year
        original_day = env.current_day
        original_start_datetime = getattr(env, '_render_start_datetime', None)

        try:
            self.reset_time_tracking()

            for t in range(final_index + 1):
                for obj, _ in time_step_snapshot:
                    try:
                        obj.time_step = t
                    except AttributeError:
                        pass

                timestamp = self.get_iso_timestamp()
                rows_by_filename[f"exported_data_community_ep{episode_num}.csv"].append(
                    {"timestamp": timestamp, **env.as_dict()}
                )

                for building in env.buildings:
                    rows_by_filename[f"exported_data_{building.name.lower()}_ep{episode_num}.csv"].append(
                        {"timestamp": timestamp, **building.as_dict()}
                    )
                    battery = building.electrical_storage
                    rows_by_filename[f"exported_data_{building.name.lower()}_battery_ep{episode_num}.csv"].append(
                        {"timestamp": timestamp, **battery.as_dict()}
                    )

                    for charger in building.electric_vehicle_chargers or []:
                        if charger not in original_charger_state:
                            original_charger_state[charger] = (
                                charger.connected_electric_vehicle,
                                charger.incoming_electric_vehicle,
                            )
                        self._set_charger_render_state(charger, t, ev_lookup)
                        rows_by_filename[f"exported_data_{building.name.lower()}_{charger.charger_id}_ep{episode_num}.csv"].append(
                            {"timestamp": timestamp, **charger.as_dict()}
                        )

                rows_by_filename[f"exported_data_pricing_ep{episode_num}.csv"].append(
                    {"timestamp": timestamp, **env.buildings[0].pricing.as_dict(t)}
                )

                for ev in env.electric_vehicles:
                    rows_by_filename[f"exported_data_{ev.name.lower()}_ep{episode_num}.csv"].append(
                        {"timestamp": timestamp, **ev.as_dict()}
                    )

        finally:
            for charger, state in original_charger_state.items():
                charger.connected_electric_vehicle, charger.incoming_electric_vehicle = state

            self.restore_render_time_step(time_step_snapshot)
            env.year = original_year
            env.current_day = original_day
            env._render_start_datetime = original_start_datetime

        for filename, rows in rows_by_filename.items():
            file_path = Path(env.new_folder_path) / filename
            if file_path.exists():
                file_path.unlink()
            self.write_render_rows(filename, rows)

    def save_to_csv(self, filename: str, data: Mapping[str, Any]):
        """Save one render row to CSV."""

        env = self.env

        if env._buffer_render and getattr(env, '_defer_render_flush', False):
            env._render_buffer[filename].append(dict(data))
            return

        self.write_render_rows(filename, [dict(data)])

    def flush_render_buffer(self):
        """Write any buffered render rows to disk."""

        env = self.env

        if not getattr(env, '_render_buffer', None):
            return

        has_pending_rows = any(env._render_buffer.values())
        if not has_pending_rows:
            env._render_buffer.clear()
            return

        try:
            target_dir = Path(env.new_folder_path)
        except Exception:
            target_dir = None

        if target_dir is not None:
            LOGGER.info("Writing buffered render exports to %s ...", target_dir)

        original_defer = env._defer_render_flush
        original_buffer_state = env._buffer_render
        env._defer_render_flush = False
        env._buffer_render = False

        try:
            for filename, rows in list(env._render_buffer.items()):
                if rows:
                    self.write_render_rows(filename, rows)
        finally:
            env._render_buffer.clear()
            env._buffer_render = original_buffer_state
            env._defer_render_flush = original_defer

    def write_render_rows(self, filename: str, rows: List[Mapping[str, Any]]):
        """Write one or more render rows to disk with minimal rewrites."""

        env = self.env
        file_path = Path(env.new_folder_path) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            return

        buffered_fieldnames = list(dict.fromkeys(field for row in rows for field in row.keys()))

        if not file_path.exists():
            fieldnames = buffered_fieldnames
            with file_path.open('w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, '') for field in fieldnames})
            return

        with file_path.open('r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []

        missing_fieldnames = [field for field in buffered_fieldnames if field not in existing_fieldnames]
        if missing_fieldnames:
            extended_fieldnames = [*existing_fieldnames, *missing_fieldnames]
            with file_path.open('w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=extended_fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, '') for field in extended_fieldnames})
                for row in rows:
                    writer.writerow({field: row.get(field, '') for field in extended_fieldnames})
            return

        with file_path.open('a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=existing_fieldnames)
            for row in rows:
                writer.writerow({field: row.get(field, '') for field in existing_fieldnames})

    def ensure_output_dir(self, *, ensure_exists: bool = True):
        """Prepare the render output directory and optionally create it on disk."""

        env = self.env
        base_render_path = Path(
            getattr(env, 'render_output_root', Path(__file__).resolve().parents[1] / 'render_logs')
        ).expanduser()

        if ensure_exists:
            try:
                base_render_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                fallback = (Path.cwd() / 'render_logs').resolve()
                fallback.mkdir(parents=True, exist_ok=True)
                env.render_output_root = fallback
                base_render_path = fallback

        render_dir = getattr(env, '_render_directory_path', None)
        needs_new_dir = render_dir is None

        if not needs_new_dir and ensure_exists:
            render_dir = Path(render_dir)
            try:
                needs_new_dir = not render_dir.is_relative_to(base_render_path)
            except AttributeError:
                needs_new_dir = base_render_path not in render_dir.parents and render_dir != base_render_path

        if needs_new_dir:
            if env.render_session_name:
                render_dir = (base_render_path / Path(env.render_session_name)).expanduser().resolve()
            else:
                if getattr(env, '_render_timestamp', None) is None:
                    env._render_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                render_dir = (base_render_path / env._render_timestamp).resolve()

            env._render_directory_path = render_dir
        else:
            render_dir = Path(env._render_directory_path)

        if ensure_exists:
            render_dir.mkdir(parents=True, exist_ok=True)
            if not env._render_dir_initialized:
                if env.render_session_name:
                    for csv_file in render_dir.glob('exported_*.csv'):
                        try:
                            csv_file.unlink()
                        except OSError:
                            pass
                env._render_dir_initialized = True

        env.new_folder_path = str(render_dir)

    def get_iso_timestamp(self) -> str:
        """Return current episode timestamp string in ISO format."""

        env = self.env

        if env.time_step == 0 or getattr(env, '_render_start_datetime', None) is None:
            self.reset_time_tracking()

        start_datetime = env._render_start_datetime
        timestamp_dt = start_datetime + datetime.timedelta(seconds=env.time_step * env.seconds_per_time_step)
        env.year = timestamp_dt.year
        env.current_day = timestamp_dt.day

        return timestamp_dt.strftime("%Y-%m-%dT%H:%M:%S")

    def override_render_time_step(self, index: int):
        """Temporarily set time_step to `index` for the environment and descendants."""

        env = self.env
        snapshot = []

        def _record(obj):
            if hasattr(obj, 'time_step'):
                snapshot.append((obj, obj.time_step))
                obj.time_step = index

        _record(env)
        for building in getattr(env, 'buildings', []):
            _record(building)
            electrical_storage = getattr(building, 'electrical_storage', None)
            if electrical_storage is not None:
                _record(electrical_storage)

            for charger in getattr(building, 'electric_vehicle_chargers', []) or []:
                _record(charger)

            for washing_machine in getattr(building, 'washing_machines', []) or []:
                _record(washing_machine)

        for ev in getattr(env, 'electric_vehicles', []):
            _record(ev)
            battery = getattr(ev, 'battery', None)
            if battery is not None:
                _record(battery)

        return snapshot

    @staticmethod
    def restore_render_time_step(snapshot):
        for obj, value in snapshot:
            try:
                obj.time_step = value
            except AttributeError:
                pass

    def reset_time_tracking(self):
        """Reset render timestamp tracking to episode start."""

        env = self.env
        start_offset = getattr(env.episode_tracker, 'episode_start_time_step', 0)
        base_datetime = datetime.datetime.combine(env.render_start_date, datetime.time())
        base_datetime += datetime.timedelta(seconds=start_offset * env.seconds_per_time_step)
        env._render_start_datetime = base_datetime
        env.year = base_datetime.year
        env.current_day = base_datetime.day
