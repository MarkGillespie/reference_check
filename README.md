# PDF Reference validity checker
This is a simple command-line tool to check references in a given PDF. The tool was scripted with VSCode and Gemini 3 Pro and proofread and tested by me.

## What does it do?
- Loads an input PDF (locally)
- Parses the reference section in the PDF for a list of references. It expects the format: `Author List. YEAR. Title of the paper. Potential other information...`
    - This may sometimes lead to false negatives! Todo: parse a .bbl file.
- Searches DBLP using the query: `[title] [author last names]`. All author names are used in the query, so that papers with incorrect authors will _not_ be matched.
- Checks the top five results to see if:
    - The paper title matches the record in DBLP.
    - The author list has at least one overlapping last name.
- If no match is found, the reference is reported `not found` and a list is printed in the console of all the papers that were not found.
- The results are also logged in a `reference_check/filename.log` file for later inspection.

## What should I do with this?
The script is expected to give false negatives. A **negative** is considered a reference that is **not found** in DBLP. That means **you have to check the flagged references yourself**.

It is unlikely to give **false positives**. If the script returns a list of `OK` results, you can likely trust the references in the paper.

## Installation
Install dependencies
```
pip install pypdf requests
```

## Running the tool
You can run the tool on a single file with:
```bash 
python reference_check.py /path/to/pdf/file.pdf
```
There are other input modes for loading files from a url or in batch mode: 

### URL
```bash
# Downloads the PDF, processes it and deletes it afterward
python reference_check.py https://path_to_pdf.org/file.pdf
```

### Batch processing
```bash
# Input a path to a folder and process each PDF file in the folder.
python reference_check.py /path/to/folder
# Input the path to a txt file with PDF urls. The txt file should have a single 
python reference_check.py url_list.txt
```

### Output
The results are printed in the console and logged in a `reference_check/filename.log` file for later inspection.

## Known issues
Most issues are due to incorrect PDF parsing. This can be fixed by parsing a .bbl file directly and I found it unnecessary to layer extra complexity on top of the script to account for these errors. Feel free to open a pull request if you find a good fix.
- A paper that uses the numbered citation system (`[1] Author names`) can lead to incorrect parsing.
- The script removes dashes from the title to account for line-breaks. This may remove dashes that should be present, leading to false negatives.
- Special characters in author names may not be matched in DBLP.
- Quotes are not parsed correctly for DBLP.