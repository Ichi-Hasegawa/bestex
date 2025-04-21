#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


# データセット内のファイル名を命名規則に従う形に変更(0906の新データ用)
def rename_files(directory, search_str, replace_str):
    for root, dirs, files in os.walk(directory):

        for file in files:

            # full path
            filename = os.path.join(root, file)

            if search_str in filename:  # ファイル名に検索対象の文字列が含まれているかチェック
                old_path = os.path.join(root, file)
                new_file_name = replace_str + file  # "NG "をファイル名に追加
                new_path = os.path.join(root, new_file_name)

                # 変更前と変更後のファイル名を表示
                print(f"変更前: {old_path}")
                print(f"変更後: {new_path}")

                # ユーザーに変更を確認させる
                user_input = input("このファイル名を変更しますか？(y/n): ").strip().lower()
                if user_input == "y":
                    os.rename(old_path, new_path)
                    print(f"ファイル名を変更しました: {new_path}")
                else:
                    print("ファイル名の変更をキャンセルしました。")

            else:
                print(f"検索対象の文字列が含まれていません: {file}")


if __name__ == "__main__":

    # 探索する基準ディレクトリ
    base_dir = "/net/nfs3/export/dataset/morita/tlo/bestex-weld2"

    # 変更する文字列
    search_str = "20240906/"
    replace_str = "NG "

    rename_files(base_dir, search_str, replace_str)
