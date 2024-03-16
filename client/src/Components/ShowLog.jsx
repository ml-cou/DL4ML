import React from "react";
import { FaAnglesRight } from "react-icons/fa6";
import AgGridTable from "./AgGridTable";

function ShowLog({ data = [], setData }) {
  return (
    <>
      <h1 className="text-center font-primary text-2xl font-semibold underline">
        Execution Log
      </h1>
      <div className="mt-4 space-y-4 bg-gray-100 shadow-md p-3 py-5 rounded">
        {data.map((val, key) => {
          return (
            <div key={key} className="flex items-start gap-3">
              <p className="">
                <FaAnglesRight size={15} />
              </p>
              <div className="space-y-4">
                {val["text"] && (
                  <p className="font-secondary text-lg text-gray-800/85 font-semibold  relative -top-2">
                    {val["text"]}
                  </p>
                )}
                {val["table"] && val["table"].length > 0 && (
                  <AgGridTable rowData={val["table"]} />
                )}
                {val["graph"] && (
                  <img src={`data:image/png;base64,${val["graph"]}`} alt="" />
                )}
              </div>
            </div>
          );
        })}
        <div className="flex justify-center">
          <button
            className="p-2 px-6 rounded bg-blue-600 text-white"
            onClick={() => setData([])}
          >
            Clear All
          </button>
        </div>
      </div>
    </>
  );
}

export default ShowLog;
