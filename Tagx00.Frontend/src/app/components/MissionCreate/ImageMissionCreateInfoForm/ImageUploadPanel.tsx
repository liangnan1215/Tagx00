import React from 'react';
import { observer } from "mobx-react";
import { observable } from "mobx";
import { Upload, Button, Icon, message } from 'antd';
import { UploadFile, UploadProps } from "antd/lib/upload/interface";

interface Props {
  onFileListChange: (fileList: UploadFile[]) => void;
  fileList: UploadFile[];
  maxFileNum: number;
  multiple: boolean;
}

function validFile(file: UploadFile) {
  return file.type.startsWith("image/");
}

export class ImageUploadPanel extends React.Component<Props, {}> {



  render() {
    const props: UploadProps = {
      action: '//jsonplaceholder.typicode.com/posts/',
      listType: "picture",
      onRemove: (file) => {
        const newList = this.props.fileList.filter(x => x !== file);
        this.props.onFileListChange(newList);
      },
      beforeUpload: (_, fileList) => {

        const validFiles = fileList.filter(validFile);
        this.props.onFileListChange([...this.props.fileList, ...validFiles]);
        return false;
      },
      fileList: this.props.fileList,
      multiple: this.props.multiple
    };

    return <div>
      <Upload {...props}>
        {this.props.fileList.length >= this.props.maxFileNum
          ? null : <Button>
            <Icon type="upload" /> Select File
          </Button>}

      </Upload>
    </div>
  }
}
