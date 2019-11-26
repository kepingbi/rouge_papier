import os
import pkg_resources
from subprocess import check_output
import re
import pandas as pd


AVG_RECALL_PATT = r"ROUGE-{} Average_R: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
AVG_PRECISION_PATT = r"ROUGE-{} Average_P: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
AVG_FMEASURE_PATT = r"ROUGE-{} Average_F: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"

def compute_rouge(config_path, show_all=True, max_ngram=4, lcs=False,
                  stemmer=True, length=100, length_unit="word",
                  number_of_samples=1000, scoring_formula="A",
                  remove_stopwords=False, return_conf=False):
    rouge_path = pkg_resources.resource_filename(
        'rouge_papier', os.path.join('data', 'ROUGE-1.5.5.pl'))
    rouge_data_path = pkg_resources.resource_filename(
        'rouge_papier', os.path.join('rouge_data'))

    #-n 4 -m -a -l 100 -x -c 95
    #-r 1000 -f A -p 0.5 -t 0
    args = ["perl", rouge_path, "-e", rouge_data_path, "-a"]

    if max_ngram > 0:
        args.extend(["-n", str(max_ngram)])

    if not lcs:
        args.append("-x")

    if show_all:
        args.append("-d")

    if stemmer:
        args.append("-m")

    if remove_stopwords:
        args.append("-s")

    if length_unit == "word":
        if length < 300:
            args.extend(["-l", str(length)])
        #when the word length is more than 300, just evaluate with all the words in the summary (full-length)
    elif length_unit == "byte":
        if length < 300:
            args.extend(["-b", str(length)])
    else:
        raise Exception(
            "length_unit must be either 'word' or 'byte' but found {}".format(
                length_unit))

    args.extend(["-r", str(number_of_samples)])

    if scoring_formula not in ["A", "B"]:
        raise Exception(
            "scoring_formula must be either 'A' or 'B' but found {}".format(
                scoring_formula))
    else:
        args.extend(["-f", scoring_formula])

    args.extend(["-z", "SPL", config_path])

    print(" ".join(args)) #print command
    output = check_output(" ".join(args), shell=True).decode("utf8")
    dataframes = []
    confs = []
    for r in range(1, max_ngram + 1):
        o, conf = convert_output(output, r)
        dataframes.append(o)
        confs.append(conf)
    if lcs:
        o, conf = convert_output(output, "L")
        dataframes.append(o)
        confs.append(conf)

    df = pd.concat(dataframes, axis=1)
    if return_conf:
        conf = pd.concat(confs, axis=0)
        return df, conf
    else:
        return df

def convert_output(output, rouge=1):
    data = []
    avg_recall_patt = AVG_RECALL_PATT.format(rouge)
    avg_precision_patt = AVG_PRECISION_PATT.format(rouge)
    avg_fmeasure_patt = AVG_FMEASURE_PATT.format(rouge)
    patt = r"ROUGE-{} Eval (.*?) R:(.*?) P:(.*?) F:(.*?)$".format(rouge)
    for match in re.findall(patt, output, flags=re.MULTILINE):
        name, recall, prec, fmeas = match
        data.append((name, float(recall), float(prec), float(fmeas)))
    match = re.search(avg_recall_patt, output, flags=re.MULTILINE)
    avg_recall = float(match.groups()[0])
    match = re.search(avg_precision_patt, output, flags=re.MULTILINE)
    avg_precision = float(match.groups()[0])
    match = re.search(avg_fmeasure_patt, output, flags=re.MULTILINE)
    avg_fmeasure = float(match.groups()[0])

    lower_conf = float(match.groups()[1])
    upper_conf = float(match.groups()[2])
    data.append(("average", avg_recall, avg_precision, avg_fmeasure))

    df = pd.DataFrame(data, columns=["name", "rouge-{}-R".format(rouge),
        "rouge-{}-P".format(rouge), "rouge-{}-F".format(rouge)])
    df.set_index("name", inplace=True)
    conf = pd.DataFrame([[lower_conf, upper_conf]],
                        columns=["95% conf. lb.", "95% conf. ub."])
    conf.index = ["rouge-{}".format(rouge)]
    #print("modified rouge")
    #print(df)
    #print(conf)
    return df, conf
