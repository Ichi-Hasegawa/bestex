#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on: Nov 29, 2024
# Author: Kentaro Nishida

from concurrent.futures import ThreadPoolExecutor
import paramiko
from rich.console import Console
from rich.progress import BarColumn, Progress
from rich.table import Table

# User Settings
servers = [
    {"hostname": "reaper"},
    {"hostname": "zeus"},
    {"hostname": "predator"},
    {"hostname": "antimon"},
    {"hostname": "triton"},
    {"hostname": "mana"},
]

# If you want to use a different username, you can specify it here.
# username = "hoge"
username = input("Enter the username for SSH connections: ")


def parse_top_output(output):
    lines = output.split("\n")
    memory_info = ""
    memory_usage = 0.0

    for line in lines:
        if "MiB Mem" in line:
            memory_info = line.strip()
            try:
                total, free, used, buff_cache = map(
                    lambda x: float(x.replace(",", "")) / 1024,
                    [x.split()[0] for x in line.split(":")[1].split(",")],
                )
                memory_usage = (used / total) * 100
                memory_info = f"{used:.2f} GB used / {total:.2f} GB total"
            except Exception:
                memory_usage = 0.0

    return memory_info, memory_usage


def get_server_info(server):
    hostname = server["hostname"]
    result = {}

    try:
        # Use the username specified by the user
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username)

        # Execute top
        stdin, stdout, stderr = client.exec_command("top -b -n 1")
        top_output = stdout.read().decode()
        memory_info, memory_usage = parse_top_output(top_output)
        result["memory"] = memory_info
        result["memory_usage"] = memory_usage

        # Execute nvidia-smi
        stdin, stdout, stderr = client.exec_command("nvidia-smi")
        result["gpu"] = stdout.read().decode()

        client.close()

    except Exception as e:
        result["error"] = str(e)

    return hostname, result


console = Console()
with ThreadPoolExecutor() as executor:
    results = list(executor.map(get_server_info, servers))

table = Table.grid(expand=True)
table.add_column(justify="center")
table.add_column(justify="center")

# Split server info into two columns
rows = [results[i : i + 2] for i in range(0, len(results), 2)]
for row in rows:
    row_tables = []
    for hostname, data in row:
        server_table = Table(title=f"[bold]{hostname}[/bold]", show_lines=True, expand=True)
        if "error" in data:
            server_table.add_row(f"[red]Error[/red]: {data['error']}")
        else:
            if data["memory_usage"] >= 75:
                color = "red"
            elif data["memory_usage"] >= 50:
                color = "yellow"
            else:
                color = "blue"

            memory_bar = Progress(
                BarColumn(bar_width=20, complete_style=color, finished_style=color),
                expand=True,
            )
            memory_task = memory_bar.add_task("", total=100, completed=data["memory_usage"])
            server_table.add_row(f"[bold]Memory[/bold]: {data['memory']}  (Memory Used : [bold]{data['memory_usage']:.2f}%[/bold])", memory_bar)
            server_table.add_row(f"[bold]GPU[/bold]:\n{data['gpu']}")
        row_tables.append(server_table)
    table.add_row(*row_tables)

console.print(table)