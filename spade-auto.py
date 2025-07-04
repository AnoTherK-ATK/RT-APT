import pexpect
import time
import json
import os

# Đường dẫn tới file log gốc và nơi lưu file tổng hợp
SOURCE_JSON_PATH = "/home/kda/log.json"
DEST_JSON_LOG = "/home/kda/collected_logs.json"

def load_existing_log():
    if os.path.exists(DEST_JSON_LOG):
        try:
            with open(DEST_JSON_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[!] Collected log file is corrupted. Starting fresh.")
    return []

def append_log_json_array(cycle_index):
    if not os.path.exists(SOURCE_JSON_PATH):
        print(f"[!] [{cycle_index}] Source log not found at {SOURCE_JSON_PATH}")
        return

    try:
        with open(SOURCE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)  # giả sử file log.json là 1 object hoặc array JSON

        collected = load_existing_log()
        collected.append(data)

        with open(DEST_JSON_LOG, "w", encoding="utf-8") as f:
            json.dump(collected, f, indent=2, ensure_ascii=False)

        print(f"[+] [{cycle_index}] Appended JSON log to collected_logs.json")

    except Exception as e:
        print(f"[!] Error during JSON append: {e}")

def run_add_remove_cycle(cycle_count=3, wait_seconds=30):
    print(f"[+] Starting spade control for {cycle_count} cycles...")
    child = pexpect.spawn("spade control", encoding='utf-8')
    child.expect("->")
    i = 1
    while(True):
        print(f"\n===== Cycle {i} / {cycle_count} =====")
        i += 1
        # Add reporter
        child.sendline("add storage JSON output=/home/kda/log.json")
        child.expect("->")
        print(f"[+] [{i}] storage added.")

        # Wait
        print(f"[*] [{i}] Waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)

        # Remove reporter
        child.sendline("remove storage JSON")
        child.expect("->")
        print(f"[+] [{i}] storage removed.")

        # Append collected log
        append_log_json_array(i)
        f = open(SOURCE_JSON_PATH, "w", encoding="utf-8")
        f.write("")
        f.close()
    child.sendline("exit")
    child.expect(pexpect.EOF)
    print("\n[+] Finished all cycles.")

if __name__ == "__main__":
    run_add_remove_cycle(cycle_count=5, wait_seconds=30)
