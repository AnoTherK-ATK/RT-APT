import string

def clean_json_file(input_path="reverse_cleaned.json", output_path=None):
    if output_path is None:
        output_path = input_path  # Ghi đè chính file cũ

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Chỉ giữ lại ký tự printable
    cleaned = ''.join(c for c in content if c in string.printable or c in '\n\t')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"[+] Cleaned file written to {output_path}")

# Chạy làm sạch
if __name__ == "__main__":
    clean_json_file("reverse_cleaned.json")
