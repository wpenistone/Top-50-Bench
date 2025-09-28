import sys
import os
import json
import re
import time
import datetime
import hashlib
import math
from itertools import product
from collections import defaultdict
import statistics

import requests
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from rapidfuzz import fuzz

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QDoubleSpinBox,
    QSpinBox,
    QFormLayout,
    QGroupBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QDockWidget,
    QProgressBar,
    QTextEdit,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QMenu,
    QComboBox,
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QCursor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

RBO_P, FUZZY_MATCH_RATIO = 0.95, 90
FIG_SIZE, FONT_SIZE, SEED = (18, 18), 9, 42
RESULTS_DIR = "results"
DARK_BACKGROUND_COLOR = "#2E2E2E"
MAX_RETRIES = 2
EXPECTED_NAME_COUNT = 50
TRIM_TOLERANCE = 5
RATE_LIMIT_WAIT = 5

USER_PROMPTS = {
    "General Knowledge": """Give me a list of your top 50, best humans currently alive right now. Give me only their full names. Use the name each individual is most commonly known by (e.g., 'Bill Gates', not 'William Henry Gates III'). Ensure all names contain only basic Latin characters (a-z, A-Z) and spaces. Ensure there are no duplicate names in the list.""",
    "Unsung Scientists": """Give me a list of your top 50 living scientists or engineers whose work has had the greatest unseen impact on daily modern life. Give me only their full names.""",
    "Open Source Figures": """Give me a list of your top 50 most influential figures in the history of the open-source software movement, living or deceased. Give me only their full names.""",
    "Controversial Thinkers": """Give me a list of your top 50 most influential but controversial living public intellectuals or thinkers. Give me only their full names.""",
    "Biotech Pioneers": """Give me a list of your top 50 living figures leading the revolution in biotechnology, genetic engineering, and longevity science. Give me only their full names.""",
}
SYSTEM_PROMPT_TEMPLATE = "You are an assistant. Fulfill the user's request. Provide the output in the specified JSON format. Ensure the list contains exactly 50 unique names."
JSON_FORMAT_PROMPT = """**Output Format:**\n{\n  "names": [\n    "Full Name 1",\n    "Full Name 2"\n  ]\n}"""

BASE_PROVIDER_COLORS = {
    "google": "#4285F4",
    "openai": "#10A37F",
    "anthropic": "#D97757",
    "meta": "#006CFF",
    "mistralai": "#FF7500",
    "xai": "#000C5B",
    "alibaba": "#FF6A00",
    "nvidia": "#76B900",
    "zhipu": "#C71585",
    "deepseek": "#FFD700",
    "microsoft": "#00A4EF",
    "meta-llama": "#006CFF",
    "mistral": "#FF7500",
    "nousresearch": "#9370DB",
    "huggingfaceh4": "#FFD700",
    "openrouter": "#8A2BE2",
}


def get_provider_from_name(model_name: str) -> str:
    return model_name.split("/")[0].lower()


def get_provider_color(provider_name: str) -> str:
    provider_name = provider_name.lower()
    if provider_name in BASE_PROVIDER_COLORS:
        return BASE_PROVIDER_COLORS[provider_name]
    hash_object = hashlib.sha256(provider_name.encode())
    hex_dig = hash_object.hexdigest()
    r, g, b = int(hex_dig[0:2], 16), int(hex_dig[2:4], 16), int(hex_dig[4:6], 16)
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\s*\(.*\)\s*", "", name).strip()
    return re.sub(r"\bsir\b", "", name).strip()


def consolidate_names(all_names: list[str], ratio_threshold: int) -> dict[str, str]:
    unique_names = sorted(list(set(all_names)), key=len, reverse=True)
    mapping = {}
    while unique_names:
        canonical_name = unique_names.pop(0)
        mapping[canonical_name] = canonical_name
        matches = [
            n for n in unique_names if fuzz.ratio(canonical_name, n) > ratio_threshold
        ]
        if matches:
            unique_names = [n for n in unique_names if n not in matches]
            for match in matches:
                mapping[match] = canonical_name
    return mapping


def rank_biased_overlap(list1: list, list2: list, p: float, idf_weights: dict) -> float:
    if not list1 or not list2:
        return 0
    s1, s2, score, weight = set(), set(), 0.0, 1.0
    for d in range(max(len(list1), len(list2))):
        if d < len(list1):
            s1.add(list1[d])
        if d < len(list2):
            s2.add(list2[d])
        intersection = s1.intersection(s2)
        overlap_score = sum(idf_weights.get(name, 1.0) for name in intersection)
        score += weight * (overlap_score / (d + 1))
        weight *= p
    return (1 - p) * score


def calculate_similarity_matrix(
    model_data: dict, p_val: float, idf_weights: dict
) -> pd.DataFrame:
    model_names = list(model_data.keys())
    similarity_matrix = pd.DataFrame(
        0.0, index=model_names, columns=model_names, dtype=float
    )
    for model_a, model_b in product(model_names, repeat=2):
        if model_a == model_b:
            sim = rank_biased_overlap(
                model_data[model_a],
                model_data[model_a],
                p=p_val,
                idf_weights=idf_weights,
            )
            similarity_matrix.loc[model_a, model_a] = sim
        elif model_a < model_b:
            sim = rank_biased_overlap(
                model_data[model_a],
                model_data[model_b],
                p=p_val,
                idf_weights=idf_weights,
            )
            similarity_matrix.loc[model_a, model_b] = sim
            similarity_matrix.loc[model_b, model_a] = sim
    return similarity_matrix


def generate_mds_figure(similarity_matrix: pd.DataFrame) -> Figure:
    if len(similarity_matrix.index) < 2:
        return None
    dissimilarity_matrix = similarity_matrix.max().max() - similarity_matrix
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=SEED,
        normalized_stress="auto",
    )
    positions_2d = mds.fit_transform(dissimilarity_matrix)
    model_names = similarity_matrix.index
    model_to_provider = {name: get_provider_from_name(name) for name in model_names}
    node_colors = [get_provider_color(model_to_provider[prov]) for prov in model_names]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.set_facecolor(DARK_BACKGROUND_COLOR)
    ax.set_facecolor(DARK_BACKGROUND_COLOR)
    ax.scatter(
        positions_2d[:, 0],
        positions_2d[:, 1],
        s=600,
        c=node_colors,
        alpha=0.9,
        edgecolors="gray",
    )
    for i, name in enumerate(model_names):
        ax.text(
            positions_2d[i, 0],
            positions_2d[i, 1],
            name,
            ha="center",
            va="center",
            color="black",
            fontsize=FONT_SIZE,
            weight="bold",
        )
    ax.set_title(
        "Model Similarity", fontsize=24, pad=20
    )
    ax.margins(0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("none")
    unique_providers = sorted(list(set(model_to_provider.values())))
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=p,
            markerfacecolor=get_provider_color(p),
            markersize=15,
        )
        for p in unique_providers
    ]
    ax.legend(
        handles=legend_handles,
        title="Model Providers",
        loc="upper right",
        facecolor="#121212",
        framealpha=0.8,
    )
    fig.tight_layout()
    return fig


TEMPERATURE = 0.95
HIGH_EFFORT_REASONING = True


class OpenRouterClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_completion(self, model_name: str, messages: list[dict]) -> list[str]:
        body = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": TEMPERATURE,
        }
        if HIGH_EFFORT_REASONING:
            body["reasoning"] = {"effort": "high"}
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            names_data = json.loads(data["choices"][0]["message"]["content"])
            if "names" in names_data and isinstance(names_data["names"], list):
                return names_data["names"]
            else:
                raise ValueError("JSON response valid, but missing 'names' list.")
        except Exception as e:
            raise ValueError(f"API Error for {model_name}: {e}")


class ExperimentWorker(QObject):
    progress_log = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_key, models, samples, delay, prompts):
        super().__init__()
        self.api_key, self.models, self.samples, self.delay, self.prompts = (
            api_key,
            models,
            samples,
            delay,
            prompts,
        )
        self.is_running = True

    def run(self):
        try:
            client = OpenRouterClient(self.api_key)
            final_data = {}
            total_steps = len(self.prompts) * len(self.models) * self.samples
            current_step = 0
            self.progress_log.emit(
                f"--- Starting Multi-Prompt Data Collection (Delay: {self.delay}s) ---"
            )

            for category, user_prompt_text in self.prompts.items():
                if not self.is_running:
                    break
                self.progress_log.emit(f"\n--- Category: {category} ---")
                raw_model_data_for_category = defaultdict(list)
                for model_name in self.models:
                    if not self.is_running:
                        break
                    for i in range(self.samples):
                        current_step += 1
                        if current_step > 1 and self.delay > 0:
                            self.progress_log.emit(f"  - Pausing for {self.delay}s...")
                            time.sleep(self.delay)
                        if not self.is_running:
                            self.progress_log.emit("Collection cancelled.")
                            return

                        self.progress_log.emit(
                            f"({current_step}/{total_steps}) Querying {model_name} (Sample {i+1}/{self.samples})..."
                        )
                        self.progress_update.emit(current_step, total_steps)

                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                            {
                                "role": "user",
                                "content": f"{user_prompt_text}\n\n{JSON_FORMAT_PROMPT}",
                            },
                        ]
                        for attempt in range(MAX_RETRIES + 1):
                            try:
                                names = client.get_completion(model_name, messages)
                                if len(names) != len(
                                    set(name.lower().strip() for name in names)
                                ):
                                    raise ValueError(
                                        "Response contained duplicate names."
                                    )
                                num_names = len(names)
                                if not (
                                    EXPECTED_NAME_COUNT
                                    <= num_names
                                    <= EXPECTED_NAME_COUNT + TRIM_TOLERANCE
                                ):
                                    raise ValueError(
                                        f"Expected ~{EXPECTED_NAME_COUNT}, received {num_names}."
                                    )
                                if num_names > EXPECTED_NAME_COUNT:
                                    names = names[:EXPECTED_NAME_COUNT]
                                raw_model_data_for_category[
                                    f"{model_name}_sample_{i}"
                                ] = names
                                self.progress_log.emit(
                                    f"  - SUCCESS: Received {len(names)} names."
                                )
                                break
                            except Exception as e:
                                if attempt < MAX_RETRIES:
                                    wait_time = (
                                        RATE_LIMIT_WAIT if "429" in str(e) else 1
                                    )
                                    self.progress_log.emit(
                                        f"  - WARN: Attempt {attempt+1} failed. Retrying in {wait_time}s... Reason: {e}"
                                    )
                                    time.sleep(wait_time)
                                else:
                                    self.progress_log.emit(
                                        f"  - ERROR: Skipping sample for {model_name} after {MAX_RETRIES+1} attempts."
                                    )

                final_data[category] = dict(raw_model_data_for_category)

            if not final_data:
                raise ValueError("No valid data could be collected for any category.")
            self.progress_log.emit("\nMulti-prompt data collection complete.")
            self.finished.emit(final_data)
        except Exception as e:
            self.error.emit(f"An error occurred during collection: {e}")

    def stop(self):
        self.is_running = False


class EditSampleDialog(QDialog):
    def __init__(self, sample_name, names_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Sample: {sample_name}")
        self.layout = QVBoxLayout(self)
        self.names_edit = QPlainTextEdit()
        self.names_edit.setPlainText("\n".join(names_list))
        self.layout.addWidget(self.names_edit)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)
        self.resize(400, 600)

    def get_names(self):
        return [
            line.strip()
            for line in self.names_edit.toPlainText().splitlines()
            if line.strip()
        ]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1600, 1000)
        self.active_run_data = None
        self.active_filepath = None
        self.is_dirty = False
        self.graph_window = None
        self._create_menu_bar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.run_tab = QWidget()
        self.browse_tab = QWidget()
        self.tabs.addTab(self.run_tab, "Run New Experiment")
        self.tabs.addTab(self.browse_tab, "Browse & Edit")
        self._setup_run_tab()
        self._setup_browse_tab()
        self._create_dockable_graph_window()
        log_group = QGroupBox("Application Log")
        log_layout = QVBoxLayout()
        self.app_log = QTextEdit()
        self.app_log.setReadOnly(True)
        log_layout.addWidget(self.app_log)
        log_group.setLayout(log_layout)
        log_group.setFixedHeight(150)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(log_group)
        self.populate_results_list()
        self.auto_load_first_or_new()
        self.update_window_title()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("&New Run", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.create_new_run)
        load_action = QAction("&Load Run File...", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_file_for_editing)
        save_action = QAction("&Save Run File", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_edited_file)
        save_as_action = QAction("Save Run File &As...", self)
        save_as_action.triggered.connect(self.save_edited_file_as)
        file_menu.addAction(new_action)
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = menu_bar.addMenu("&View")
        self.pop_out_graph_action = QAction("Pop Out Map Window", self, checkable=True)
        self.pop_out_graph_action.triggered.connect(self.toggle_graph_popout)
        view_menu.addAction(self.pop_out_graph_action)
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def _create_dockable_graph_window(self):
        self.graph_dock = QDockWidget("Similarity Map", self)
        self.graph_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.graph_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.graph_canvas = FigureCanvas(Figure(figsize=(12, 12)))
        self.graph_dock.setWidget(self.graph_canvas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.graph_dock)
        self.clear_graph()

    def _setup_run_tab(self):

        layout = QHBoxLayout(self.run_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)
        config_group = QGroupBox("Experiment Configuration")
        config_form_layout = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(os.environ.get("OPENROUTER_API_KEY", ""))
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setRange(1, 20)
        self.samples_spinbox.setValue(1)
        self.delay_spinbox = QDoubleSpinBox()
        self.delay_spinbox.setRange(0.0, 60.0)
        self.delay_spinbox.setValue(2.0)
        self.delay_spinbox.setSingleStep(0.5)
        config_form_layout.addRow("OpenRouter API Key:", self.api_key_input)
        config_form_layout.addRow("Samples per Model:", self.samples_spinbox)
        config_form_layout.addRow("Delay (s):", self.delay_spinbox)
        config_group.setLayout(config_form_layout)
        prompt_group = QGroupBox("Prompt Category Selection")
        prompt_layout = QVBoxLayout()
        self.prompt_list = QListWidget()
        self.prompt_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        for cat in USER_PROMPTS.keys():
            item = QListWidgetItem(cat)
            self.prompt_list.addItem(item)
            item.setSelected(True)
        prompt_layout.addWidget(self.prompt_list)
        prompt_group.setLayout(prompt_layout)
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        common_models = [
            "x-ai/grok-4-fast:free",
            "deepseek/deepseek-chat-v3.1:free",
            "z-ai/glm-4.5-air:free",
            "deepseek/deepseek-r1-0528:free",
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-r1:free",
            "microsoft/mai-ds-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-4-maverick:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "google/gemma-3-27b-it:free",
            "moonshotai/kimi-dev-72b:free",
            "meta-llama/llama-4-scout:free",
        ]
        for m in sorted(common_models):
            self.model_list.addItem(QListWidgetItem(m))
        model_layout.addWidget(self.model_list)
        add_model_layout = QHBoxLayout()
        self.new_model_input = QLineEdit()
        self.add_model_button = QPushButton("Add")
        add_model_layout.addWidget(self.new_model_input)
        add_model_layout.addWidget(self.add_model_button)
        model_layout.addLayout(add_model_layout)
        self.remove_model_button = QPushButton("Remove Selected")
        model_layout.addWidget(self.remove_model_button)
        model_group.setLayout(model_layout)
        self.add_model_button.clicked.connect(self.add_model)
        self.new_model_input.returnPressed.connect(self.add_model)
        self.remove_model_button.clicked.connect(self.remove_selected_models)
        config_layout.addWidget(config_group)
        config_layout.addWidget(prompt_group)
        config_layout.addWidget(model_group)
        exec_container = QWidget()
        exec_layout = QVBoxLayout(exec_container)
        exec_control_group = QGroupBox("Execution Control")
        exec_control_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Experiment & Add to Current File")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        self.run_button.clicked.connect(self.start_experiment)
        self.stop_button.clicked.connect(self.stop_experiment)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        exec_control_layout.addLayout(button_layout)
        exec_control_layout.addWidget(self.progress_bar)
        exec_control_group.setLayout(exec_control_layout)
        run_log_group = QGroupBox("Current Run Log")
        run_log_layout = QVBoxLayout()
        self.run_log = QTextEdit()
        self.run_log.setReadOnly(True)
        run_log_layout.addWidget(self.run_log)
        run_log_group.setLayout(run_log_layout)
        exec_layout.addWidget(exec_control_group)
        exec_layout.addWidget(run_log_group)
        splitter.addWidget(config_container)
        splitter.addWidget(exec_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _setup_browse_tab(self):
        layout = QHBoxLayout(self.browse_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        results_group = QGroupBox("Saved Result Files")
        results_layout = QVBoxLayout()
        self.results_list = QListWidget()
        self.results_list.currentItemChanged.connect(
            self.load_selected_result_for_browsing
        )
        results_layout.addWidget(self.results_list)
        results_group.setLayout(results_layout)
        analysis_params_group = QGroupBox("Analysis Tuning (Auto-updates map)")
        analysis_params_layout = QFormLayout()
        self.browse_ratio_spinbox = QSpinBox()
        self.browse_ratio_spinbox.setRange(1, 100)
        self.browse_ratio_spinbox.setValue(FUZZY_MATCH_RATIO)
        self.browse_p_val_spinbox = QDoubleSpinBox()
        self.browse_p_val_spinbox.setRange(0.01, 1.0)
        self.browse_p_val_spinbox.setSingleStep(0.01)
        self.browse_p_val_spinbox.setValue(RBO_P)
        analysis_params_layout.addRow(
            "Fuzzy Match Ratio (%):", self.browse_ratio_spinbox
        )
        analysis_params_layout.addRow("RBO Persistence (p):", self.browse_p_val_spinbox)
        analysis_params_group.setLayout(analysis_params_layout)
        self.browse_ratio_spinbox.valueChanged.connect(self.regenerate_graph)
        self.browse_p_val_spinbox.valueChanged.connect(self.regenerate_graph)

        add_sample_group = QGroupBox("Add New Sample")
        add_sample_layout = QFormLayout()
        self.add_category_combo = QComboBox()
        self.add_category_combo.setEditable(True)
        self.add_model_name_edit = QLineEdit()
        self.add_model_name_edit.setPlaceholderText("provider/model-name")
        self.add_sample_num_spinbox = QSpinBox()
        self.add_sample_num_spinbox.setRange(0, 999)
        self.add_names_paste_edit = QPlainTextEdit()
        self.add_names_paste_edit.setPlaceholderText(
            "Paste raw list of names here, one per line."
        )
        add_sample_button = QPushButton("Add Sample to Current Run")
        add_sample_button.clicked.connect(self.add_sample_manually)
        add_sample_layout.addRow("Category:", self.add_category_combo)
        add_sample_layout.addRow("Model Name:", self.add_model_name_edit)
        add_sample_layout.addRow("Sample Number:", self.add_sample_num_spinbox)
        add_sample_layout.addRow("Names List:", self.add_names_paste_edit)
        add_sample_layout.addRow(add_sample_button)
        add_sample_group.setLayout(add_sample_layout)
        self.add_category_combo.currentTextChanged.connect(
            self.update_next_sample_number
        )
        self.add_model_name_edit.textChanged.connect(self.update_next_sample_number)

        left_layout.addWidget(results_group, 2)
        left_layout.addWidget(analysis_params_group, 1)
        left_layout.addWidget(add_sample_group, 1)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        data_splitter = QSplitter(Qt.Orientation.Vertical)
        metadata_group = QGroupBox("Experiment Metadata")
        metadata_layout = QVBoxLayout()
        self.metadata_viewer = QPlainTextEdit()
        self.metadata_viewer.setReadOnly(True)
        metadata_layout.addWidget(self.metadata_viewer)
        metadata_group.setLayout(metadata_layout)
        data_group = QGroupBox("Current Data (Right-click to edit/delete)")
        data_layout = QVBoxLayout()
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabels(["Category / Model / Sample", "Info"])
        data_layout.addWidget(self.data_tree)
        data_group.setLayout(data_layout)
        self.data_tree.itemDoubleClicked.connect(self.handle_item_double_click)
        self.data_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_tree.customContextMenuRequested.connect(self.show_tree_context_menu)

        data_splitter.addWidget(metadata_group)
        data_splitter.addWidget(data_group)
        right_layout.addWidget(data_splitter)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    def _initial_layout_fix(self):
        self.resizeDocks(
            [self.graph_dock], [int(self.width() * 0.5)], Qt.Orientation.Horizontal
        )

    def on_tab_changed(self, index):
        if self.tabs.widget(index) == self.browse_tab:
            self.populate_results_list()

    def perform_analysis(self, data_by_category, ratio, p_val):

        self.log("--- Starting Analysis ---")
        all_similarity_matrices = []
        model_names = set()
        for category, raw_model_data in data_by_category.items():
            if not raw_model_data:
                continue
            self.log(f"Processing category: {category}...")
            grouped_samples = defaultdict(list)
            [
                grouped_samples[k.split("_sample_")[0]].append(v)
                for k, v in raw_model_data.items()
            ]
            aggregated_model_data = {}
            master_name_list = []
            for model, samples_list in grouped_samples.items():
                model_names.add(model)
                all_model_names = set()
                for names in samples_list:
                    c_names = [normalize_name(h) for h in names if h]
                    master_name_list.extend(c_names)
                    for name in c_names:
                        all_model_names.add(name)
                name_ranks = defaultdict(list)
                for name in all_model_names:
                    for names in samples_list:
                        c_names = [normalize_name(h) for h in names if h]
                        try:
                            name_ranks[name].append(c_names.index(name) + 1)
                        except ValueError:
                            name_ranks[name].append(EXPECTED_NAME_COUNT * 2)
                median_scores = {
                    name: statistics.median(ranks) for name, ranks in name_ranks.items()
                }
                if median_scores:
                    aggregated_model_data[model] = sorted(
                        median_scores.keys(),
                        key=lambda name: (median_scores[name], name),
                    )
            if not aggregated_model_data:
                continue
            canonical_map = consolidate_names(master_name_list, ratio)
            final_model_data = {
                model: [
                    canonical_map.get(n, n)
                    for n in names
                    if not (
                        canonical_map.get(n, n) in seen
                        or seen.add(canonical_map.get(n, n))
                    )
                ]
                for model, names in aggregated_model_data.items()
                for seen in [set()]
            }
            if len(final_model_data) < 2:
                continue
            N = len(final_model_data)
            name_doc_freq = defaultdict(int)
            for names_list in final_model_data.values():
                for name in set(names_list):
                    name_doc_freq[name] += 1
            idf_weights = {
                name: 1 + math.log(N / freq) for name, freq in name_doc_freq.items()
            }
            similarity_matrix = calculate_similarity_matrix(
                final_model_data, p_val, idf_weights
            )
            min_val = similarity_matrix.min().min()
            max_val = similarity_matrix.max().max()
            if max_val > min_val:
                normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)
                all_similarity_matrices.append(normalized_matrix)
                self.log(f"Category '{category}' processed and normalized.")
        if not all_similarity_matrices:
            raise ValueError("Could not generate a similarity matrix for any category.")
        model_names = sorted(list(model_names))
        combined_matrix = pd.DataFrame(0.0, index=model_names, columns=model_names)
        for matrix in all_similarity_matrices:
            combined_matrix = combined_matrix.add(
                matrix.reindex(index=model_names, columns=model_names).fillna(0),
                fill_value=0,
            )
        self.log(
            f"Combined {len(all_similarity_matrices)} matrices into a final holistic matrix."
        )
        self.log(f"Final Combined Similarity Matrix:\n{combined_matrix.round(3)}")
        fig = generate_mds_figure(combined_matrix)
        if fig is None:
            raise ValueError("Final map could not be generated.")
        self.log("analysis complete.")
        return fig

    def start_experiment(self):
        selected_models = [item.text() for item in self.model_list.selectedItems()]
        selected_prompts = {
            item.text(): USER_PROMPTS[item.text()]
            for item in self.prompt_list.selectedItems()
        }
        if not self.api_key_input.text():
            QMessageBox.warning(
                self, "API Key Missing", "Please enter your OpenRouter API key."
            )
            return
        if not selected_models:
            QMessageBox.warning(self, "No Models", "Please select at least one model.")
            return
        if not selected_prompts:
            QMessageBox.warning(
                self, "No Prompts", "Please select at least one prompt category."
            )
            return
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.clear_graph()
        self.run_log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.thread = QThread()
        self.worker = ExperimentWorker(
            api_key=self.api_key_input.text(),
            models=selected_models,
            samples=self.samples_spinbox.value(),
            delay=self.delay_spinbox.value(),
            prompts=selected_prompts,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_experiment_data_received)
        self.worker.error.connect(self.on_error)
        self.worker.progress_log.connect(self.run_log.append)
        self.worker.progress_update.connect(self.update_progress_bar)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(
            lambda: (
                self.run_button.setEnabled(True),
                self.stop_button.setEnabled(False),
                self.progress_bar.setFormat("Finished"),
            )
        )
        self.thread.start()

    def on_experiment_data_received(self, new_data_by_category):
        self.log("Experiment finished. Merging collected data into current run.")
        if not self.active_run_data:
            self.create_new_run()
        if "data_by_category" not in self.active_run_data:
            self.active_run_data["data_by_category"] = {}
        for category, cat_data in new_data_by_category.items():
            if category not in self.active_run_data["data_by_category"]:
                self.active_run_data["data_by_category"][category] = {}
            for sample_key, names_list in cat_data.items():
                self.active_run_data["data_by_category"][category][
                    sample_key
                ] = names_list
        self.set_dirty(True)
        self.update_all_views()
        self.save_edited_file()
        self.tabs.setCurrentWidget(self.browse_tab)

    def regenerate_graph(self):
        if not self.active_run_data or "data_by_category" not in self.active_run_data:
            return
        try:
            fig = self.perform_analysis(
                data_by_category=self.active_run_data["data_by_category"],
                ratio=self.browse_ratio_spinbox.value(),
                p_val=self.browse_p_val_spinbox.value(),
            )
            self.display_graph(fig)
        except Exception as e:
            self.log(f"ERROR re-generating map: {e}")
            self.clear_graph()

    def populate_browse_views(self):
        self.metadata_viewer.clear()
        self.data_tree.clear()
        if not self.active_run_data:
            return
        self.metadata_viewer.setPlainText(
            json.dumps(self.active_run_data.get("metadata", {}), indent=4)
        )
        self.add_category_combo.clear()
        data_to_display = self.active_run_data.get("data_by_category", {})
        if data_to_display:
            self.add_category_combo.addItems(sorted(data_to_display.keys()))

        for category, model_data in sorted(data_to_display.items()):
            cat_item = QTreeWidgetItem(self.data_tree, [category])
            cat_item.setToolTip(0, "Category Level")

            samples_by_model = defaultdict(list)
            for sample_key, names_list in model_data.items():
                try:
                    model_name, _ = sample_key.rsplit("_sample_", 1)
                    samples_by_model[model_name].append((sample_key, names_list))
                except ValueError:
                    continue

            for model_name, samples in sorted(samples_by_model.items()):
                model_item = QTreeWidgetItem(cat_item, [model_name])
                model_item.setToolTip(0, f"Model Level\nCategory: {category}")
                for sample_key, names_list in sorted(
                    samples, key=lambda x: int(x[0].rsplit("_", 1)[1])
                ):
                    _, sample_num_str = sample_key.rsplit("_sample_", 1)
                    sample_item = QTreeWidgetItem(
                        model_item,
                        [
                            f"Sample {int(sample_num_str) + 1}",
                            f"{len(names_list)} names",
                        ],
                    )
                    sample_item.setData(0, Qt.ItemDataRole.UserRole, sample_key)
                    sample_item.setToolTip(0, f"Sample Level\nKey: {sample_key}")

        self.data_tree.expandToDepth(1)
        self.data_tree.resizeColumnToContents(0)

    def handle_item_double_click(self, item, column):
        if item.childCount() == 0:
            self.edit_sample(item)

    def show_tree_context_menu(self, position):
        item = self.data_tree.itemAt(position)
        if not item:
            return
        menu = QMenu()
        level = 0
        if item.parent():
            level = 1
            if item.parent().parent():
                level = 2

        if level == 2:
            edit_action = menu.addAction("Edit Sample")
            edit_action.triggered.connect(lambda: self.edit_sample(item))
            menu.addSeparator()
            delete_action = menu.addAction("Delete Sample")
            delete_action.triggered.connect(lambda: self.delete_sample(item))
        elif level == 1:
            delete_action = menu.addAction("Delete Model")
            delete_action.triggered.connect(lambda: self.delete_model(item))
        elif level == 0:
            delete_action = menu.addAction("Delete Category")
            delete_action.triggered.connect(lambda: self.delete_category(item))

        menu.exec(self.data_tree.mapToGlobal(position))

    def edit_sample(self, item):
        sample_key = item.data(0, Qt.ItemDataRole.UserRole)
        category = item.parent().parent().text(0)
        names = self.active_run_data["data_by_category"][category].get(sample_key, [])
        dialog = EditSampleDialog(sample_key, names, self)
        if dialog.exec():
            new_names = dialog.get_names()
            self.active_run_data["data_by_category"][category][sample_key] = new_names
            self.log(f"Updated sample '{sample_key}' in memory.")
            self.set_dirty(True)
            self.update_all_views()

    def delete_sample(self, item):
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete this sample?\n({item.data(0, Qt.ItemDataRole.UserRole)})",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            sample_key = item.data(0, Qt.ItemDataRole.UserRole)
            category = item.parent().parent().text(0)
            del self.active_run_data["data_by_category"][category][sample_key]
            self.log(f"Deleted sample '{sample_key}'.")
            self.set_dirty(True)
            self.update_all_views()

    def delete_model(self, item):
        category = item.parent().text(0)
        model_name = item.text(0)
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete model '{model_name}' and all its samples from category '{category}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            keys_to_delete = [
                k
                for k in self.active_run_data["data_by_category"][category]
                if k.startswith(f"{model_name}_sample_")
            ]
            for key in keys_to_delete:
                del self.active_run_data["data_by_category"][category][key]
            self.log(f"Deleted model '{model_name}' from '{category}'.")
            self.set_dirty(True)
            self.update_all_views()

    def delete_category(self, item):
        category = item.text(0)
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the entire category '{category}' and all its data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self.active_run_data["data_by_category"][category]
            self.log(f"Deleted category '{category}'.")
            self.set_dirty(True)
            self.update_all_views()

    def add_sample_manually(self):
        category = self.add_category_combo.currentText().strip()
        model_name = self.add_model_name_edit.text().strip()
        sample_num = self.add_sample_num_spinbox.value()
        names = [
            line.strip()
            for line in self.add_names_paste_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if not all([category, model_name, names]):
            QMessageBox.warning(
                self,
                "Input Error",
                "Category, Model Name, and a list of names are all required.",
            )
            return

        sample_key = f"{model_name}_sample_{sample_num}"
        if (
            category in self.active_run_data.get("data_by_category", {})
            and sample_key in self.active_run_data["data_by_category"][category]
        ):
            if (
                QMessageBox.question(
                    self,
                    "Overwrite Sample",
                    f"'{sample_key}' already exists in '{category}'. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.No
            ):
                return

        if "data_by_category" not in self.active_run_data:
            self.active_run_data["data_by_category"] = {}
        if category not in self.active_run_data["data_by_category"]:
            self.active_run_data["data_by_category"][category] = {}

        self.active_run_data["data_by_category"][category][sample_key] = names
        self.log(f"Manually added/updated '{sample_key}' in category '{category}'.")
        self.set_dirty(True)
        self.update_all_views()
        self.add_names_paste_edit.clear()
        self.add_model_name_edit.clear()

    def update_next_sample_number(self):
        category = self.add_category_combo.currentText().strip()
        model_name = self.add_model_name_edit.text().strip()
        if not self.active_run_data or not category or not model_name:
            self.add_sample_num_spinbox.setValue(0)
            return

        cat_data = self.active_run_data.get("data_by_category", {}).get(category, {})
        max_sample = -1
        for key in cat_data.keys():
            if key.startswith(model_name + "_sample_"):
                try:
                    max_sample = max(max_sample, int(key.rsplit("_", 1)[1]))
                except (ValueError, IndexError):
                    continue
        self.add_sample_num_spinbox.setValue(max_sample + 1)

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About Cognitive Cartographer",
            "Cognitive Cartographer v7.1\n\nA tool for creating and editing holistic, multi-prompt MDS maps of LLM similarity.",
        )

    def auto_load_first_or_new(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")], reverse=True
        )
        if files:
            self.load_file_for_editing(os.path.join(RESULTS_DIR, files[0]), True)
        else:
            self.create_new_run()

    def create_new_run(self):
        if self.is_dirty and not self.prompt_to_save():
            return
        self.active_run_data = {
            "metadata": {
                "source": "New Run",
                "run_date": datetime.datetime.now().isoformat(),
            },
            "data_by_category": {},
        }
        self.active_filepath = None
        self.set_dirty(True)
        self.update_all_views()
        self.tabs.setCurrentWidget(self.browse_tab)

    def load_file_for_editing(self, filepath=None, on_startup=False):
        if not on_startup and self.is_dirty and not self.prompt_to_save():
            return
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load Run File", RESULTS_DIR, "JSON Files (*.json)"
            )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.active_run_data = json.load(f)
            if (
                "raw_model_data" in self.active_run_data
                and "data_by_category" not in self.active_run_data
            ):
                self.active_run_data["data_by_category"] = {
                    "General Knowledge": self.active_run_data.pop("raw_model_data")
                }
                self.log("Converted old file format to new multi-prompt format.")
                self.set_dirty(True)
            self.active_filepath = filepath
            self.set_dirty(False)
            self.update_all_views()
            self.log(f"Loaded '{os.path.basename(filepath)}'.")
        except Exception as e:
            self.log(f"ERROR loading {filepath}: {e}")

    def save_edited_file(self):
        if not self.active_filepath:
            self.save_edited_file_as()
            return
        if not self.active_run_data:
            self.log("ERROR: No data to save.")
            return
        with open(self.active_filepath, "w", encoding="utf-8") as f:
            json.dump(self.active_run_data, f, indent=4)
        self.set_dirty(False)
        self.log(f"Saved changes to '{os.path.basename(self.active_filepath)}'.")
        self.populate_results_list()

    def save_edited_file_as(self):
        if not self.active_run_data:
            self.log("ERROR: No data to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Run File As", RESULTS_DIR, "JSON Files (*.json)"
        )
        if filepath:
            self.active_filepath = filepath
            self.save_edited_file()

    def update_all_views(self):
        self.populate_browse_views()
        self.regenerate_graph()

    def set_dirty(self, is_dirty):
        self.is_dirty = is_dirty
        self.update_window_title()

    def update_window_title(self):
        title = "Cognitive Cartographer v7.1"
        filename = (
            os.path.basename(self.active_filepath)
            if self.active_filepath
            else "Untitled Run"
        )
        self.setWindowTitle(f"{title} - {filename}{'*' if self.is_dirty else ''}")

    def populate_results_list(self):
        current_selection = (
            os.path.basename(self.active_filepath) if self.active_filepath else None
        )
        self.results_list.clear()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")], reverse=True
        )
        for f in files:
            self.results_list.addItem(QListWidgetItem(f))
        if current_selection:
            items = self.results_list.findItems(
                current_selection, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.results_list.setCurrentItem(items[0])

    def load_selected_result_for_browsing(self, current_item, previous_item):
        if not current_item:
            return
        self.load_file_for_editing(os.path.join(RESULTS_DIR, current_item.text()))

    def toggle_graph_popout(self, checked):
        if checked:
            if not self.graph_window:
                self.graph_window = QMainWindow(self)
                self.graph_window.setWindowTitle("Graph Viewer")
                self.graph_window.closing.connect(self.return_graph_to_dock)
            self.graph_dock.setWidget(None)
            self.graph_dock.setVisible(False)
            self.graph_window.setCentralWidget(self.graph_canvas)
            self.graph_window.show()
        elif self.graph_window and self.graph_window.isVisible():
            self.graph_window.close()

    def return_graph_to_dock(self):
        if self.graph_window:
            self.graph_window.setCentralWidget(None)
            self.graph_window = None
        self.graph_dock.setWidget(self.graph_canvas)
        self.graph_dock.setVisible(True)
        if self.pop_out_graph_action.isChecked():
            self.pop_out_graph_action.setChecked(False)

    def add_model(self):
        model_name = self.new_model_input.text().strip()
        if model_name and not self.model_list.findItems(
            model_name, Qt.MatchFlag.MatchExactly
        ):
            self.model_list.addItem(QListWidgetItem(model_name))
        self.new_model_input.clear()

    def remove_selected_models(self):
        for item in self.model_list.selectedItems():
            self.model_list.takeItem(self.model_list.row(item))

    def stop_experiment(self):
        if hasattr(self, "worker"):
            self.worker.stop()
            self.log("Sending stop signal...")
            self.stop_button.setEnabled(False)

    def on_error(self, message):
        self.log(f"ERROR: {message}")
        self.run_log.append(f"FATAL ERROR: {message}")
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.quit()
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setFormat("Error")

    def update_progress_bar(self, value, total):
        percent = int((value / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"%p ({value}/{total})")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.app_log.append(f"[{timestamp}] {msg}")

    def clear_graph(self):
        self.graph_canvas.figure.clear()
        self.graph_canvas.figure.set_facecolor(DARK_BACKGROUND_COLOR)
        ax = self.graph_canvas.figure.add_subplot(111)
        ax.set_facecolor(DARK_BACKGROUND_COLOR)
        ax.tick_params(axis="x", colors="none")
        ax.tick_params(axis="y", colors="none")
        for spine in ax.spines.values():
            spine.set_edgecolor("none")
            self.graph_canvas.draw()

    def display_graph(self, fig):
        if not fig:
            self.clear_graph()
            return
        old_fig = self.graph_canvas.figure
        self.graph_canvas.figure = fig
        if old_fig and old_fig is not fig:
            plt.close(old_fig)
        try:
            self.graph_canvas.figure.tight_layout()
        except Exception as e:
            self.log(f"Warning: Could not apply tight_layout. Error: {e}")
        self.graph_canvas.draw_idle()

    def prompt_to_save(self) -> bool:
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self.save_edited_file()
            return not self.is_dirty
        return reply == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if self.graph_window:
            self.graph_window.close()
        if self.is_dirty and not self.prompt_to_save():
            event.ignore()
            return
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window._initial_layout_fix)
    sys.exit(app.exec())
