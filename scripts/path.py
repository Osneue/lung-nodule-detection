# src/util/path.py
import os
import sys

def ensure_project_root():
    current_file = os.path.abspath(__file__)
    # 注意这里往上两层才能到项目根（src/util）
    project_root = os.path.dirname(os.path.dirname(current_file))
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, project_root)
    sys.path.insert(1, src_path)
    if os.getcwd() != project_root:
        print(f"[INFO] 切换工作目录到项目根目录: {project_root}")
        os.chdir(project_root)
    # show module searching path
    #print(sys.path)

if __name__ == '__main__':
    ensure_project_root()