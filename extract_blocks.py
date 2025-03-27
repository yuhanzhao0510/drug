import json
import os
from datetime import datetime


def save_json_fields(json_data: dict, base_folder: str = "output"):
    """
    将JSON数据按字段分割保存为JSON格式文件

    Args:
        json_data: JSON数据字典
        base_folder: 基础输出文件夹路径
    """

    # 创建以drugbank_id命名的文件夹
    folder_path = os.path.join(base_folder, 'blocks')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    # 获取当前时间和用户信息
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = os.getenv('USER', 'yuhanzhao0510')

    # 为每个字段创建单独的JSON文件
    for field, value in json_data.items():
        # 构建文件路径，使用.json扩展名
        file_path = os.path.join(folder_path, f"{field}.json")

        try:
            # 创建包含元数据的字典
            field_data = {
                "field" : field,
                "content": value
            }

            # 保存为JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(field_data, f, indent=2, ensure_ascii=False)
            print(f"Created file: {file_path}")

        except Exception as e:
            print(f"Error saving {field}: {str(e)}")


def main():
    # 读取JSON数据
    input_file = "A01AA01/A01AA01.json"  # 你的输入文件名
    output_folder = "A01AA01"  # 输出文件夹

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 保存字段到单独的JSON文件
        save_json_fields(data, output_folder)
        print("\nProcessing completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'!")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()