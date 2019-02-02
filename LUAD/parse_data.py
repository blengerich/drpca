import numpy as np
import json
import subprocess
import csv
import gzip


def read_data(directory, f_name):
    transcript_names = []
    levels = []
    with gzip.open(directory + '/' + f_name, 'rb') as f:
        content = f.read()

    for row in content.split(b'\n'):
        parts = row.split(b'\t')
        if len(parts) < 2:
            continue
        transcript_names.append(str(parts[0]))
        levels.append(float(parts[1]))

    return transcript_names, levels


def get_data_for_file_id(f_id, f_name):
    # Navigate to the file
    directory = "Data/" + f_id
    return read_data(directory, f_name)


def get_data_from_dirs():
    read_all_data = True

    cases = []
    cases_stages = []
    cases_pfi = []
    controls = []
    cases_case_ids = []
    controls_case_ids = []
    case_ids_set = set([])
    matches = []
    expected_transcript_names = None

    with open("../pfi.json", 'r') as pfi_file:
        pfi_dict = json.load(pfi_file)

    metadata_file = "metadata.cart.2017-11-28T22_15_20.317932.json"
    with open(metadata_file, 'r') as json_file:
        head = json.load(json_file)
    for i, entry in enumerate(head):

        assert len(entry["cases"]) == 1
        file_id = entry["file_id"]
        file_name = entry["file_name"]
        if "htseq" in file_name:
            continue
        print("{}\t{}".format(i, file_name))

        case = entry["cases"][0]
        case_id = case["case_id"]


        assert len(case["samples"]) == 1
        sample = case["samples"][0]

        if sample["sample_type"] == "Primary Tumor":
            if case_id in case_ids_set:
                print("Found Match")
                try:
                    matches.append([controls_case_ids.index(case_id), len(cases_case_ids)])
                except:
                    print("Tumor matches Tumor id")
            case_ids_set.add(case_id)
            cases_case_ids.append(case_id)
        elif sample["sample_type"] == "Solid Tissue Normal":
            if case_id in case_ids_set:
                print("Found Match")
                try:
                    matches.append([len(controls_case_ids), cases_case_ids.index(case_id)])
                except:
                    print("Control matches Control id")
            case_ids_set.add(case_id)
            controls_case_ids.append(case_id)

        if read_all_data:
            transcript_names, levels = get_data_for_file_id(file_id, file_name)
            if expected_transcript_names is not None:
                assert transcript_names == expected_transcript_names
            else:
                expected_transcript_names = transcript_names

            if sample["sample_type"] == "Primary Tumor":
                cases.append(levels)
                try:
                    cases_stages.append(case["diagnoses"][0]["tumor_stage"])
                except KeyError:
                    cases_stages.append("None")
                case_pfi = []
                try:
                    for x in pfi_dict[case["submitter_id"]]:
                        try:
                            case_pfi.append(int(x))
                        except ValueError:
                            case_pfi.append(-1)
                except KeyError:
                    case_pfi = [-1, -1]
                cases_pfi.append(case_pfi)
            elif sample["sample_type"] == "Solid Tissue Normal":
                controls.append(levels)
            else:
                print(sample["sample_type"])

    cases = np.array(cases)
    controls = np.array(controls)
    print("Case shape:   {},\nControl shape:{}.".format(cases.shape, controls.shape))
    print(matches)
    return cases, controls, np.array(matches), expected_transcript_names, cases_pfi, cases_stages

cases, controls, matches, transcript_names, cases_pfi, cases_stages = get_data_from_dirs()
with open('matches.npy', 'wb') as npy_out:
    np.save(npy_out, matches)

with open('transcript_names.npy', 'wb') as npy_out:
    np.save(npy_out, np.array(transcript_names))

with open("cases.npy", 'wb') as npy_out:
    np.save(npy_out, cases)

with open("controls.npy", 'wb') as npy_out:
    np.save(npy_out, controls)

np.save("cases_pfi", cases_pfi)
np.save("cases_stages", cases_stages)

"""
def gunzip(directory, f_name):
    # Unzip the File
    command = "gunzip {}/{}.gz".format(directory, f_name)
    print(command)
    subprocess.call([command])
"""