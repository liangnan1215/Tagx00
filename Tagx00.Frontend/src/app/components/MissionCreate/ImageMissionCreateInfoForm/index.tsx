import React from 'react';
import { observer } from "mobx-react";
import { action, observable, runInAction, toJS } from "mobx";
import { ImageMissionCreateInfo } from "./ImageMissionCreateInfo";
import { Form, Input, Icon, Checkbox, Button, Modal } from 'antd';
import { FormItemProps } from "antd/lib/form/FormItem";
import { ImageMissionType } from "../../../models/mission/image/ImageMission";
import { ImageUploadPanel } from "./ImageUploadPanel";
import { requesterService } from "../../../api/RequesterService";
const FormItem = Form.Item;
const CheckboxGroup = Checkbox.Group;

interface Props {
  token: string;
}


const imageMissionTypes = Object.keys(ImageMissionType)
  .map(x => ({ label: x, value: x}));

function formItemProps(valid: boolean, error: string): FormItemProps  {
  return {
    validateStatus: valid ? "success" : "error",
    help: valid? null : error
  };
}

@observer
export class ImageMissionCreateInfoForm extends React.Component<Props, {}> {

  @observable info: ImageMissionCreateInfo = new ImageMissionCreateInfo();
  @observable uploading = false;

  @action onTitleChange = (e) => {
    this.info.title = e.target.value;
  };

  @action onDescriptionChange = (e) => {
    this.info.description = e.target.value;
  };


  @action onTypeChange = (checkedValues: string[]) => {
    this.info.imageMissionTypes = checkedValues.map(x => ImageMissionType[x]);
  };

  @action onFileListChange= (files) => {
    this.info.images = files;
  };

  @action onCoverImageChange = (files) => {
    this.info.coverImage = files[0];
  };

  @action upload = async () => {
    console.log(this.info.images);

  };

  submit = async () => {
    console.log(toJS(this.info));

    const {token ,id} = await requesterService.createMission(this.info.missionCreateVo, this.props.token);

    console.log(token, id);

    // upload cover

    runInAction(() => { this.uploading =true;});
    const coverFormData = new FormData();
    coverFormData.append("files[]", this.info.coverImage as any);

    const coverUrl = await requesterService.uploadImageFile(id, coverFormData, 1, true, token);

    for (let i =0;i<this.info.images.length;i++) {
      const imageFormData = new FormData();
      imageFormData.append("files[]", this.info.images[i] as any);
      const img = await requesterService.uploadImageFile(id, imageFormData, i+2, false, token);
      console.log(img);
    }

    Modal.success({
      title: '上传成功',
      content: '任务创建成功',
    });

    runInAction(() => this.uploading = false);
  };


  render() {

    return <Form className="login-form">
      <FormItem {...formItemProps(this.info.titleValid, "请输入标题")}>
        <Input onChange={this.onTitleChange}
               placeholder={"标题"}
               value={this.info.title}
        />
      </FormItem>
      <FormItem {...formItemProps(this.info.descriptionValid, "请输入描述")}>
        <Input onChange={this.onDescriptionChange}
               placeholder={"描述"}
               value={this.info.description}
        />
      </FormItem>
      <FormItem>
        <CheckboxGroup options={imageMissionTypes}
                       value={this.info.imageMissionTypes}
                       onChange={this.onTypeChange}/>
      </FormItem>
      <p>上传一张封面图</p>
      <ImageUploadPanel onFileListChange={this.onCoverImageChange}
                        fileList={[this.info.coverImage].filter(x => !!x)}
                        maxFileNum={1}
                        multiple={false}
      />
      <p>所有图片</p>
      <ImageUploadPanel onFileListChange={this.onFileListChange}
                        fileList={this.info.images}
                        maxFileNum={Number.MAX_SAFE_INTEGER}
                        multiple={true}
      />
      <Button type={"primary"} onClick={this.submit} loading={this.uploading}>
        确认
      </Button>
    </Form>
  }
}
