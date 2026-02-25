import re
import os
import sys

# Add directory to sys path
sys.path.append(os.path.abspath("CodeAgent"))
from codeagent_libs import search_code, find_file, view_file_content

def test_xml_parsing():
    resp_content = """
    I need to look for auth keys.
    <search_code>import requests</search_code>
    
    And find the auth module:
    <find_file>auth.py</find_file>
    
    And check some lines:
    <view_file>
      <filepath>CodeAgent/codeagent_libs.py</filepath>
      <start_line>200</start_line>
      <end_line>205</end_line>
    </view_file>
    """
    
    tool_results = []
    for match in re.finditer(r'<search_code>(.*?)</search_code>', resp_content, re.DOTALL):
        query = match.group(1).strip()
        res = search_code(query)
        tool_results.append(f"Result for <search_code>{query}</search_code>:\n{res}")
        
    for match in re.finditer(r'<find_file>(.*?)</find_file>', resp_content, re.DOTALL):
        pattern = match.group(1).strip()
        res = find_file(pattern)
        tool_results.append(f"Result for <find_file>{pattern}</find_file>:\n{res}")
        
    for match in re.finditer(r'<view_file>\s*<filepath>(.*?)</filepath>\s*<start_line>(\d+)</start_line>\s*<end_line>(\d+)</end_line>\s*</view_file>', resp_content, re.DOTALL):
        fpath = match.group(1).strip()
        s_line = int(match.group(2))
        e_line = int(match.group(3))
        res = view_file_content(fpath, s_line, e_line)
        tool_results.append(f"Result for <view_file> {fpath} ({s_line}-{e_line}):\n{res}")

    print("--- PARSED RESULTS ---")
    for tr in tool_results:
        print(tr)
        print("-" * 20)

if __name__ == "__main__":
    test_xml_parsing()
