import argparse
import datetime
import json
import os
import pyspark
import random
import sys
from io import BytesIO
import base64
from typing import IO, Any, List
import math
import pickle
from contextlib import contextmanager
from data_pipeline_utils import TiktokDataPipeline, TiktokDataFusedPipeline
import subprocess
import threading


# HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console hdfs'
HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'
INPUT_DIR = ["hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/input_meta"]
OUTPUT_DIR = ["hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/data"]
MAX_THREAD_NUM = 30
OUTPUT_PARTITION_NUM = 10000
PARTITION_NUM = 1000
DATE_FORMAT = '%Y%m%d'


def hlist(file):
    files = []
    pipe = subprocess.Popen("{} dfs -ls {}".format(HADOOP_BIN, file), shell=True,
                            stdout=subprocess.PIPE)
    for line in pipe.stdout:  
        line = line.strip()
        # drwxr-xr-x   - user group  4 file
        if len(line.split()) < 5:
            continue
        if '_SUCCESS' in line.split()[-1].decode("utf8"):
            continue
        files.append(line.split()[-1].decode("utf8"))
    pipe.stdout.close()  # type: ignore
    pipe.wait()
    return files


def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    # is_hdfs = hdfs_path.startswith('hdfs')
    # if is_hdfs:
    #     return hdfs_open(hdfs_path, mode)
    # else:
    #     return open(hdfs_path, mode)

    return hdfs_open(hdfs_path, mode)


@contextmanager  # type: ignore
def hdfs_open(hdfs_path: str, mode: str = "r") -> IO[Any]:
    """
        打开一个 hdfs 文件, 用 contextmanager.

        Args:
            hfdfs_path (str): hdfs文件路径
            mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hexists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path)) == 0
    return os.path.exists(file_path)


def create_data_fetch_fn():
    def data_fetch_worker(data_list):
        thread_num = len(data_list)
        results = [None for _ in range(len(data_list))]
        # data_pipeline = TiktokDataPipeline()
        data_pipeline = TiktokDataFusedPipeline()

        def fetch_one_data(index, data):
            data_item = json.loads(data)
            item_id = data_item['item_id']
            try:
                video_meta = data_pipeline.request_all_info(item_id, frame_num=8)
                if video_meta is None:
                    results[index] = None
                else:
                    if data_item is not None:
                        video_meta.update(data_item)

                    results[index] = json.dumps(video_meta)
            except Exception as e:
                print(f"fail to get item data for Item {item_id}\n, the exception is {e}")
                results[index] = None

        threads = []
        for i in range(thread_num):
            # create a new thread and add it to the list
            thread = threading.Thread(target=fetch_one_data, args=(i, data_list[i]))
            threads.append(thread)

        # start all the threads
        for thread in threads:
            thread.start()

        # wait for all the threads to finish
        for thread in threads:
            thread.join()
        
        return results
    
    return data_fetch_worker


def gen_formatted_instance(x):
    if len(x) > 3:
        return json.dumps([x[0], x[1], x[2], x[3]])
    else:
        return json.dumps([x[0], x[1], x[2]])


def save(args, instances, output_path, repartition=0):
    # Saving to HDFS
    print("save_to_hdfs")
    #instances=instances.map(gen_formatted_instance) #.values()
    if repartition:
        instances = instances.repartition(repartition)
    instances.saveAsTextFile(output_path)


def main(args):
    spark_conf = pyspark.SparkConf()
    sc = pyspark.SparkContext(conf=spark_conf)

    date_str = args.data_date.strftime(DATE_FORMAT)
    all_count = 0
    for input_dir, output_dir in zip(INPUT_DIR, OUTPUT_DIR):
        total_count = 0
        input_file_paths = hlist(input_dir)
        output_file_paths = [f"{output_dir}/{os.path.split(file)[-1].split('.')[0]}" for file in input_file_paths]

        for input_path, output_path in zip(input_file_paths, output_file_paths):
            print(input_path)
            if input_path.find("_SUCCESS") >= 0 or input_path.find("_temporary") >= 0:
                print(input_path, "omitted")
                continue
            # try:
            #cur_part = f.split('/')[-1].split('.')[0]
            #if hexists(os.path.join(output_path, cur_part, "_SUCCESS")):
            #    print(f, "existed")
            #    continue
            os.system('/opt/tiger/yarn_deploy/hadoop/bin/hadoop fs -mkdir -p %s' % output_path)

            rdd = sc.textFile(input_path)
            num_elements = rdd.count()
            # num_partitions = math.ceil(num_elements / MAX_THREAD_NUM)
            rdd = rdd.zipWithIndex().map(lambda x: (x[1] // MAX_THREAD_NUM, x[0]))
            rdd = rdd.groupByKey().mapValues(list).map(lambda x:x[1])
            rdd = rdd.repartition(PARTITION_NUM)

            data_fetch_worker_fn = create_data_fetch_fn()
            rdd = rdd.map(data_fetch_worker_fn)
            rdd = rdd.flatMap(lambda x: x).filter(lambda x: x is not None)
            # rdd = rdd.glom().flatMap(lambda x: [(i,) for i in x]).glom()  # make each partition only have one element
            
            print("$$$$$$$$$$$")
            print(f"total {rdd.count()} items")
            os.system('/opt/tiger/yarn_deploy/hadoop/bin/hadoop fs -rm -r %s' % output_path)
            # save(args, rdd, out, repartition=OUTPUT_PARTITION_NUM)
            print("saving to hdfs")

            rdd = rdd.repartition(OUTPUT_PARTITION_NUM)
            rdd.saveAsTextFile(output_path)

            # except Exception as e:
            #     print(f, e)

        #print(f"{input_path} total count: {total_count}")
        #all_count += total_count
    
    #print(f"all count: {all_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gandalf instance converter')
    args = parser.parse_args()

    data_date = datetime.datetime.strptime('${date}', DATE_FORMAT)
    print("time is: ",data_date)
    setattr(args, 'data_date', data_date)
    main(args)