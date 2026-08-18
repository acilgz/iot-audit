
import os
import re
import argparse

def extract_tflite_from_header(header_file_path, output_model_path):
    with open(header_file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'alignas\(8\) const unsigned char g_mlp_model\[\] = \{([^}]+)\};', content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find the model data in the header file")
    
    hex_data = match.group(1)
    
    hex_values = []
    for line in hex_data.split(','):
        line = line.strip()
        if line:
            if line.startswith('0x'):
                hex_values.append(int(line, 16))
            else:
                try:
                    hex_values.append(int(line, 16))
                except ValueError:
                    hex_values.append(int(line))
    
    with open(output_model_path, 'wb') as f:
        f.write(bytes(hex_values))
    
    print(f"Successfully converted {header_file_path} to {output_model_path}")
    print(f"Model size: {len(hex_values)} bytes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default="ids_hw/ids_esp32_mlp/mlp.h")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--model-name", default="mlp_int8")
    args = ap.parse_args()

    outfile = os.path.join(args.outdir, "models", args.model_name, "model.tflite")
    
    if not os.path.exists(args.header):
        print(f"Error: Header file {args.header} not found")
        return
    
    try:
        extract_tflite_from_header(args.header, outfile)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()