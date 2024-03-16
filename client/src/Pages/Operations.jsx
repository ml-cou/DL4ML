import { UploadOutlined } from "@ant-design/icons";
import { Button, Radio, message } from "antd";
import TextArea from "antd/es/input/TextArea";
import Upload from "antd/es/upload/Upload";
import React, { useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import { BiText } from "react-icons/bi";
import { FaRegFileAudio } from "react-icons/fa6";
import AudioInput from "../Components/AudioInput";
import ShowLog from "../Components/ShowLog";

/*
CREATE ESTIMATOR salaryPredictor TYPE LR FORMULA $salary~years$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];
USE 'data/salarydb.db';
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;
*/

function Operations() {
  const [type, setType] = useState("text");
  const [audioTranscript, setAudioTranscript] = useState("");
  const [query, setQuery] = useState("");
  const [rowData, setRowData] = useState();
  const [fileList, setFileList] = useState([]);
  const [data, setData] = useState([
    // {
    //   text: "Selected lol",
    //   table: [
    //     { name: "ahnaf", age: 20 },
    //     { name: "shifat", age: 19 },
    //   ],
    // },
  ]);

  const handleExecute = async () => {
    setRowData();
    if (fileList.length === 0) {
      message.error("File is missing");
      return;
    }
    if (!query) {
      toast.error("Query can't be empty");
      return;
    }
    if (query[query.length - 1] !== ";") {
      toast.error("Invalid Query.");
      return;
    }
    let inputs = query.split(";");
    inputs = inputs.slice(0, inputs.length - 1);
    inputs = inputs.map((val) => val.trim() + ";");
    for (let input of inputs) {
      if (!input) {
        toast.error("Invalid Query.");
        return;
      }
    }
    try {
      const formData = new FormData();
      formData.append("input", inputs);
      formData.append("file", fileList[0].originFileObj);

      const res = await fetch("http://localhost:8000/test_url/", {
        method: "POST",
        body: formData,
      });
      const d = await res.json();
      console.log(d);
      setData((prev) => [...prev, d]);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="mt-10 max-w-2xl mx-auto">
      <Toaster />
      <div className="text-center">
        <Radio.Group
          size="large"
          value={type}
          onChange={(e) => setType(e.target.value)}
          className="font-semibold"
          buttonStyle="solid"
        >
          <Radio.Button value={"text"} className=" !font-secondary">
            <div className="flex items-center gap-2">
              <span className="">Use Text</span>{" "}
              <span>
                <BiText size={22} />
              </span>
            </div>
          </Radio.Button>
          <Radio.Button value={"audio"} className=" !font-secondary">
            <div className="flex items-center gap-2">
              <span>Use Audio</span>{" "}
              <span>
                <FaRegFileAudio size={22} />
              </span>
            </div>
          </Radio.Button>
        </Radio.Group>
      </div>
      {type === "audio" && (
        <AudioInput
          audioTranscript={audioTranscript}
          setAudioTranscript={setAudioTranscript}
        />
      )}
      <div className="mt-2  grid sticky top-0 bg-white z-50 py-4 ">
        <h1 className="text-left font-secondary text-lg font-semibold mb-2 ">
          Enter your query:
        </h1>
        <TextArea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={5}
          placeholder="Enter your SQL query"
          className="font-secondary text-gray-800"
        />
        <div className="mt-4">
          <Upload
            className="!text-xl"
            fileList={fileList.map((file) => ({ ...file, status: "done" }))}
            beforeUpload={(file) => {
              setFileList([{ uid: file.uid, name: file.name, status: "done" }]);
              return false;
            }}
            onChange={(e) => setFileList(e.fileList)}
          >
            {fileList.length === 0 && (
              <Button icon={<UploadOutlined />}>Upload File</Button>
            )}
          </Upload>
        </div>
        <button
          className="mt-4 w-24 ml-auto bg-blue-500 rounded text-white p-2 px-4 font-secondary font-semibold"
          onClick={handleExecute}
        >
          Execute
        </button>
      </div>
      {data.length > 0 && <ShowLog data={data} setData={setData} />}
    </div>
  );
}

export default Operations;

/*

*/
