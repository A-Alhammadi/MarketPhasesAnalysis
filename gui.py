import tkinter as tk
from tkinter import messagebox
import ast

import config
import main

class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Config GUI")

        # We'll make a frame that can scroll if you have many fields
        # (optional). For simplicity, we put them all directly in root here.
        row_idx = 0

        # 1) START_DATE
        tk.Label(self.root, text="START_DATE:").grid(row=row_idx, column=0, sticky="e")
        self.start_date_var = tk.StringVar(value=str(config.START_DATE))
        tk.Entry(self.root, textvariable=self.start_date_var, width=30).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 2) END_DATE
        tk.Label(self.root, text="END_DATE:").grid(row=row_idx, column=0, sticky="e")
        self.end_date_var = tk.StringVar(value=str(config.END_DATE))
        tk.Entry(self.root, textvariable=self.end_date_var, width=30).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 3) PHASE_CHANGES_FILE
        tk.Label(self.root, text="PHASE_CHANGES_FILE:").grid(row=row_idx, column=0, sticky="e")
        self.phase_file_var = tk.StringVar(value=config.PHASE_CHANGES_FILE)
        tk.Entry(self.root, textvariable=self.phase_file_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 4) PRICE_SMA_CHANGES_FILE
        tk.Label(self.root, text="PRICE_SMA_CHANGES_FILE:").grid(row=row_idx, column=0, sticky="e")
        self.price_file_var = tk.StringVar(value=config.PRICE_SMA_CHANGES_FILE)
        tk.Entry(self.root, textvariable=self.price_file_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 5) EXTREME_VOLUME_Z_THRESHOLD
        tk.Label(self.root, text="EXTREME_VOLUME_Z_THRESHOLD:").grid(row=row_idx, column=0, sticky="e")
        self.z_thresh_var = tk.StringVar(value=str(config.EXTREME_VOLUME_Z_THRESHOLD))
        tk.Entry(self.root, textvariable=self.z_thresh_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 6) RUN_SECTOR_ANALYSIS (bool)
        self.run_sector_var = tk.BooleanVar(value=config.RUN_SECTOR_ANALYSIS)
        tk.Checkbutton(
            self.root,
            text="RUN_SECTOR_ANALYSIS",
            variable=self.run_sector_var
        ).grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        row_idx += 1

        # 7) SECTOR_TICKERS (list -> comma separated)
        tk.Label(self.root, text="SECTOR_TICKERS (comma-separated):").grid(row=row_idx, column=0, sticky="e")
        self.sector_str_var = tk.StringVar(value=", ".join(config.SECTOR_TICKERS))
        tk.Entry(self.root, textvariable=self.sector_str_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 8) PERIODS (dict)
        tk.Label(self.root, text="PERIODS (dict):").grid(row=row_idx, column=0, sticky="e")
        self.periods_str_var = tk.StringVar(value=str(config.PERIODS))
        tk.Entry(self.root, textvariable=self.periods_str_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 9) SP500_TICKER
        tk.Label(self.root, text="SP500_TICKER:").grid(row=row_idx, column=0, sticky="e")
        self.sp500_var = tk.StringVar(value=config.SP500_TICKER)
        tk.Entry(self.root, textvariable=self.sp500_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 10) ROLLING_WINDOW
        tk.Label(self.root, text="ROLLING_WINDOW:").grid(row=row_idx, column=0, sticky="e")
        self.rolling_win_var = tk.StringVar(value=str(config.ROLLING_WINDOW))
        tk.Entry(self.root, textvariable=self.rolling_win_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 11) MA_SHORT
        tk.Label(self.root, text="MA_SHORT:").grid(row=row_idx, column=0, sticky="e")
        self.ma_short_var = tk.StringVar(value=str(config.MA_SHORT))
        tk.Entry(self.root, textvariable=self.ma_short_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 12) MA_LONG
        tk.Label(self.root, text="MA_LONG:").grid(row=row_idx, column=0, sticky="e")
        self.ma_long_var = tk.StringVar(value=str(config.MA_LONG))
        tk.Entry(self.root, textvariable=self.ma_long_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 13) NEWHIGHLOW_LOOKBACK
        tk.Label(self.root, text="NEWHIGHLOW_LOOKBACK:").grid(row=row_idx, column=0, sticky="e")
        self.nh_nl_var = tk.StringVar(value=str(config.NEWHIGHLOW_LOOKBACK))
        tk.Entry(self.root, textvariable=self.nh_nl_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 14) PERCENTABOVE_MA_WINDOW
        tk.Label(self.root, text="PERCENTABOVE_MA_WINDOW:").grid(row=row_idx, column=0, sticky="e")
        self.pct_above_var = tk.StringVar(value=str(config.PERCENTABOVE_MA_WINDOW))
        tk.Entry(self.root, textvariable=self.pct_above_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 15) RESULTS_DIR
        tk.Label(self.root, text="RESULTS_DIR:").grid(row=row_idx, column=0, sticky="e")
        self.results_dir_var = tk.StringVar(value=config.RESULTS_DIR)
        tk.Entry(self.root, textvariable=self.results_dir_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 16) PHASE_PLOT_INTERVAL
        tk.Label(self.root, text="PHASE_PLOT_INTERVAL:").grid(row=row_idx, column=0, sticky="e")
        self.phase_plot_var = tk.StringVar(value=str(config.PHASE_PLOT_INTERVAL))
        tk.Entry(self.root, textvariable=self.phase_plot_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 17) INDICATOR_PLOT_INTERVAL
        tk.Label(self.root, text="INDICATOR_PLOT_INTERVAL:").grid(row=row_idx, column=0, sticky="e")
        self.ind_plot_var = tk.StringVar(value=str(config.INDICATOR_PLOT_INTERVAL))
        tk.Entry(self.root, textvariable=self.ind_plot_var, width=10).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
        row_idx += 1

        # 18) PLOT_PHASES (dict)
        tk.Label(self.root, text="PLOT_PHASES (dict):").grid(row=row_idx, column=0, sticky="e")
        self.plot_phases_var = tk.StringVar(value=str(config.PLOT_PHASES))
        tk.Entry(self.root, textvariable=self.plot_phases_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # 19) PLOT_INDICATORS (dict)
        tk.Label(self.root, text="PLOT_INDICATORS (dict):").grid(row=row_idx, column=0, sticky="e")
        self.plot_inds_var = tk.StringVar(value=str(config.PLOT_INDICATORS))
        tk.Entry(self.root, textvariable=self.plot_inds_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
        row_idx += 1

        # Buttons (Save & Run)
        tk.Button(self.root, text="Save Changes", command=self.save_changes).grid(row=row_idx, column=0, pady=10)
        tk.Button(self.root, text="Run Main Program", command=self.run_main).grid(row=row_idx, column=1, pady=10)
        row_idx += 1

    def save_changes(self):
        """
        Save changes in memory to config.py variables. 
        """
        # START_DATE
        start_val = self.start_date_var.get().strip()
        config.START_DATE = start_val if start_val else None

        # END_DATE
        end_val = self.end_date_var.get().strip()
        if end_val.lower() == "none":
            config.END_DATE = None
        else:
            config.END_DATE = end_val if end_val else None

        # PHASE_CHANGES_FILE
        config.PHASE_CHANGES_FILE = self.phase_file_var.get().strip() or config.PHASE_CHANGES_FILE

        # PRICE_SMA_CHANGES_FILE
        config.PRICE_SMA_CHANGES_FILE = self.price_file_var.get().strip() or config.PRICE_SMA_CHANGES_FILE

        # EXTREME_VOLUME_Z_THRESHOLD
        try:
            config.EXTREME_VOLUME_Z_THRESHOLD = float(self.z_thresh_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid float for EXTREME_VOLUME_Z_THRESHOLD; reverting.")

        # RUN_SECTOR_ANALYSIS
        config.RUN_SECTOR_ANALYSIS = self.run_sector_var.get()

        # SECTOR_TICKERS
        sector_text = self.sector_str_var.get().strip()
        if sector_text:
            config.SECTOR_TICKERS = [x.strip() for x in sector_text.split(",") if x.strip()]

        # PERIODS (dict)
        periods_text = self.periods_str_var.get().strip()
        if periods_text:
            try:
                maybe_dict = ast.literal_eval(periods_text)
                if isinstance(maybe_dict, dict):
                    config.PERIODS = maybe_dict
                else:
                    messagebox.showerror("Error", "PERIODS must be a dict.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not parse PERIODS: {e}")

        # SP500_TICKER
        config.SP500_TICKER = self.sp500_var.get().strip() or config.SP500_TICKER

        # ROLLING_WINDOW
        roll_val = self.rolling_win_var.get().strip()
        if roll_val.isdigit():
            config.ROLLING_WINDOW = int(roll_val)

        # MA_SHORT
        ma_s_val = self.ma_short_var.get().strip()
        if ma_s_val.isdigit():
            config.MA_SHORT = int(ma_s_val)

        # MA_LONG
        ma_l_val = self.ma_long_var.get().strip()
        if ma_l_val.isdigit():
            config.MA_LONG = int(ma_l_val)

        # NEWHIGHLOW_LOOKBACK
        nh_nl_val = self.nh_nl_var.get().strip()
        if nh_nl_val.isdigit():
            config.NEWHIGHLOW_LOOKBACK = int(nh_nl_val)

        # PERCENTABOVE_MA_WINDOW
        pct_ab_val = self.pct_above_var.get().strip()
        if pct_ab_val.isdigit():
            config.PERCENTABOVE_MA_WINDOW = int(pct_ab_val)

        # RESULTS_DIR
        config.RESULTS_DIR = self.results_dir_var.get().strip() or config.RESULTS_DIR

        # PHASE_PLOT_INTERVAL
        config.PHASE_PLOT_INTERVAL = self.phase_plot_var.get().strip() or config.PHASE_PLOT_INTERVAL

        # INDICATOR_PLOT_INTERVAL
        config.INDICATOR_PLOT_INTERVAL = self.ind_plot_var.get().strip() or config.INDICATOR_PLOT_INTERVAL

        # PLOT_PHASES (dict)
        plot_phases_text = self.plot_phases_var.get().strip()
        if plot_phases_text:
            try:
                maybe_dict = ast.literal_eval(plot_phases_text)
                if isinstance(maybe_dict, dict):
                    config.PLOT_PHASES = maybe_dict
                else:
                    messagebox.showerror("Error", "PLOT_PHASES must be a dict.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not parse PLOT_PHASES: {e}")

        # PLOT_INDICATORS (dict)
        plot_inds_text = self.plot_inds_var.get().strip()
        if plot_inds_text:
            try:
                maybe_dict = ast.literal_eval(plot_inds_text)
                if isinstance(maybe_dict, dict):
                    config.PLOT_INDICATORS = maybe_dict
                else:
                    messagebox.showerror("Error", "PLOT_INDICATORS must be a dict.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not parse PLOT_INDICATORS: {e}")

        messagebox.showinfo("Saved", "Config has been updated in memory.")

    def run_main(self):
        """
        Runs the main program (main.main()) with current in-memory config.
        """
        answer = messagebox.askyesno("Run Main?", "Are you sure you want to run the main program?")
        if not answer:
            return
        try:
            main.main()
            messagebox.showinfo("Done", "Main program finished.")
        except Exception as e:
            messagebox.showerror("Error", f"An exception occurred in main:\n{e}")


def launch_config_gui():
    root = tk.Tk()
    gui = ConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_config_gui()
