import argparse
import curses
import os
import shutil

import yaml
from tabulate import tabulate
from utils import logs_dir

LOGS_DIR = os.path.join(logs_dir, "tensorboard")


def list_runs():
    """List all runs."""
    runs = [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))]
    return runs


def flatten_dict(d, parent_key=""):
    """Helper function to flatten nested dictionaries."""
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key))
        else:
            items.append((new_key, value))
    return items


def tabular_hyperparameters(dict_):
    """Formats hyperparameters in a tabular format."""

    # Flatten the dictionary into a list of key-value pairs
    flattened = flatten_dict(dict_)

    # Use tabulate to format the table
    table = tabulate(flattened, headers=["Parameter", "Value"], tablefmt="plain")
    return table.splitlines()  # Split into lines for curses rendering


def show_hyperparameters(run_name):
    """Shows hyperparameters of a specific run."""
    run_path = os.path.join(LOGS_DIR, run_name)
    hyperparameters_file = os.path.join(run_path, "hyper_parameters.yaml")
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, "r") as f:
            hyperparameters = yaml.safe_load(f)
        return tabular_hyperparameters(hyperparameters)
    else:
        return [f"Run '{hyperparameters_file}' doesn't exist."]


def delete_run(run_name):
    """Deletes a specific run."""
    run_path = os.path.join(LOGS_DIR, run_name)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
        return f"Deleted run '{run_name}'."
    else:
        return f"Run '{run_name}' doesn't exist."


class TensorBoardCLI:
    def __init__(self):
        self.runs = list_runs()
        self.current_index = 0
        self.current_page = 0
        self.MENU_LINES = 5

    def get_page_size(self, stdscr: curses.window) -> int:
        """Get the size of the page."""
        height, _ = stdscr.getmaxyx()
        return height - self.MENU_LINES

    def display_hyperparameters(self, stdscr: curses.window, run_name):
        """Display hyperparameters with paging."""
        hyperparameters = show_hyperparameters(run_name)
        current_index = 0
        current_page = 0
        page_size = self.get_page_size(stdscr)
        total_pages = (len(hyperparameters) + page_size - 1) // page_size

        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, f"--- Hyperparameters of run '{run_name}' ---")
            start_index = current_page * page_size
            end_index = min(start_index + page_size, len(hyperparameters))

            for i in range(start_index, end_index):
                if i == current_index:
                    stdscr.addstr(
                        i - start_index + 1,
                        0,
                        f"> {hyperparameters[i]}",
                    )
                else:
                    stdscr.addstr(i - start_index + 1, 0, f"  {hyperparameters[i]}")

            stdscr.addstr(
                page_size + 2,
                0,
                "Press 'n' for next page, 'p' for previous, or 'esc' to return.",
            )
            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP and current_index > 0:
                current_index -= 1
                if current_index < current_page * page_size:
                    current_page -= 1
            elif key == curses.KEY_DOWN and current_index < len(hyperparameters) - 1:
                current_index += 1
                if current_index == (current_page + 1) * page_size:
                    current_page += 1
            elif (
                key == ord("n") or key == curses.KEY_RIGHT
            ) and current_page < total_pages - 1:
                current_page += 1
                current_index = current_page * page_size
            elif (key == ord("p") or key == curses.KEY_LEFT) and current_page > 0:
                current_page -= 1
                current_index = current_page * page_size
            elif key == 27:  # ESC key
                break

    def display_menu(self, stdscr: curses.window):
        """Display the main menu and handle user input."""
        curses.curs_set(0)  # Hide the cursor
        page_size = self.get_page_size(stdscr)
        total_pages = (len(self.runs) + page_size - 1) // page_size

        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, "--- TensorBoard Run Manager ---")
            stdscr.addstr(1, 0, "Select an action:")
            stdscr.addstr(2, 0, "1. Show Run Hyperparameters")
            stdscr.addstr(3, 0, "2. Delete Run")
            stdscr.addstr(4, 0, "Press 'esc' to exit and navigate via the arrows.")

            # Display runs
            start_index = self.current_page * page_size
            end_index = min(start_index + page_size, len(self.runs))
            for i in range(start_index, end_index):
                if i == self.current_index:
                    stdscr.addstr(
                        i - start_index + 5, 0, f"> {self.runs[i]}", curses.A_REVERSE
                    )
                else:
                    stdscr.addstr(i - start_index + 5, 0, f"  {self.runs[i]}")

            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP and self.current_index > 0:
                self.current_index -= 1
                if self.current_index < self.current_page * page_size:
                    self.current_page -= 1
            elif key == curses.KEY_DOWN and self.current_index < len(self.runs) - 1:
                self.current_index += 1
                if self.current_index == (self.current_page + 1) * page_size:
                    self.current_page += 1
            elif (
                key == ord("n") or key == curses.KEY_RIGHT
            ) and self.current_page < total_pages - 1:
                self.current_page += 1
                self.current_index = self.current_page * page_size
            elif (key == ord("p") or key == curses.KEY_LEFT) and self.current_page > 0:
                self.current_page -= 1
                self.current_index = self.current_page * page_size
            elif key == 27:  # ESC key
                break
            elif key == ord("1") or key == 10:
                run_name = self.runs[self.current_index]
                self.display_hyperparameters(stdscr, run_name)
            elif key == ord("2") or key == 330:
                run_name = self.runs[self.current_index]
                result = delete_run(run_name)
                stdscr.clear()
                stdscr.addstr(0, 0, result)
                stdscr.addstr(2, 0, "Press any key to return to menu.")
                stdscr.refresh()
                stdscr.getch()
                self.runs = list_runs()  # Refresh the list of runs
                self.current_index = 0  # Reset index
                self.current_page = 0  # Reset page


def main():
    parser = argparse.ArgumentParser(description="Manage TensorBoard runs.")
    args = parser.parse_args()
    cli = TensorBoardCLI()
    curses.wrapper(cli.display_menu)


if __name__ == "__main__":
    main()
