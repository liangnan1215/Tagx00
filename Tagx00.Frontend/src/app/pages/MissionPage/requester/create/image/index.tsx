import React, { ReactNode } from 'react';
import { observer } from "mobx-react";
import { action, observable, runInAction, toJS } from "mobx";
import { CreditStatus, ImageMissionCreateInfo } from "./ImageMissionCreateInfo";
import { Checkbox, DatePicker, Form, Input, Modal, Button, Card, Row, Col } from 'antd';
import { FormItemProps } from "antd/lib/form/FormItem";
import { ImageMissionType } from "../../../../../models/mission/image/ImageMission";
import { ImageUploadPanel } from "./ImageUploadPanel";
import { RequesterService } from "../../../../../api/RequesterService";
import { Inject } from "react.di";
import { LocaleStore } from "../../../../../stores/LocaleStore";
import { Link } from 'react-router-dom';
import { RouterStore } from "../../../../../stores/RouterStore";
import moment from 'moment';
import { TagSelector } from "../../../../../components/TagSelector";
import { TopicService } from "../../../../../api/TopicService";
import { AsyncComponent, ObserverAsyncComponent } from "../../../../../router/AsyncComponent";
import { Loading } from "../../../../../components/Common/Loading";
import { FormItem } from "../../../../../components/Form/FormItem";
import { RichFormItem } from "../../../../../components/Form/RichFormItem";
import { CurrentCreditsIndicator } from "../../../../../components/Pay/CurrentCreditsIndicator";
import { LocaleMessage } from "../../../../../internationalization/components";
import { PayService } from "../../../../../api/PayService";
import { CreditInput } from "../../../../../components/Pay/CreditInput";
import { TopicSelector } from "./TopicSelector";
import { MissionTypeSelectPanel } from "../MissionTypeSelectPanel";
import { UserStore } from "../../../../../stores/UserStore";

const CheckboxGroup = Checkbox.Group;
const {RangePicker} = DatePicker;

interface Props {

}

const ID_PREFIX = "missions.createMission.";



/**
 *       "fields": {
        "title": "任务标题",
        "requireTitle": "请输入任务标题",
        "description": "任务描述",
        "requireDescription": "请输入任务描述",
        "topics": "主题词",
        "availableTopics": "可选主题词",
        "startDate": "开始时间",
        "requireStartDate": "请选择任务开始时间",
        "endDate": "结束时间",
        "requireEndDate": "请选择任务结束时间",
        "cover": "封面图",
        "selectFile": "选择文件",
        "IMAGE": {
          "type": {
            "DISTRICT": "区域",
            "WHOLE":"整体",
            "PART": "局部"
          },
          "requireImage": "请上传至少一张图片"
        }
      },
 */

@observer
export default class ImageMissionCreateInfoForm extends React.Component<Props, {}> {

  @observable info: ImageMissionCreateInfo = new ImageMissionCreateInfo();
  @observable uploading = false;

  @Inject localeStore: LocaleStore;
  @Inject routerStore: RouterStore;
  @Inject requesterService: RequesterService;
  @Inject topicService: TopicService;
  @Inject payService: PayService;
  @Inject userStore: UserStore;



  @action onTypeChange = (checkedValues: string[]) => {
    this.info.imageMissionTypes = checkedValues.map(x => ImageMissionType[x]);
  };


  @action onFileListChange = (files) => {
    this.info.images = files;
  };


  @action upload = async () => {
    console.log(this.info.images);

  };


  @action onTagsChange = (tags: string[]) => {
    this.info.allowedTags = tags;
  };

  @action submit = async () => {

    this.info.createAttempted = true;
    if (!this.info.valid) {
      return;
    }

    console.log(this.info.missionCreateVo);

    const {token, id} = await this.requesterService.createMission(this.info.missionCreateVo);

    console.log(token, id);

    // upload cover

    runInAction(() => {
      this.uploading = true;
    });
    const coverFormData = new FormData();
    coverFormData.append("files[]", this.info.coverImage as any);

    const coverUrl = await this.requesterService.uploadImageFile(id, coverFormData, 1, true);

    for (let i = 0; i < this.info.images.length; i++) {
      const imageFormData = new FormData();
      imageFormData.append("files[]", this.info.images[i] as any);
      const img = await this.requesterService.uploadImageFile(id, imageFormData, i + 2, false);
      console.log(img);
    }

    const modalIdPrefix = ID_PREFIX + "completeCreation.";

    const modal = Modal.success({
      title: this.localeStore.get(modalIdPrefix + "title"),
      content: this.localeStore.get(modalIdPrefix + "description", {
        to: <a onClick={() => {
          modal.destroy();
          this.routerStore.jumpTo(`/mission?missionId=${id}`);
        }}>
          {this.localeStore.get(modalIdPrefix + "to")}
        </a>
      }),
    });

    runInAction(() => this.uploading = false);
  };

  @action onAllowCustomTagChanged = (e) => {
    this.info.allowCustomTag = e.target.checked;
  };


  render() {
    const locale: any = new Proxy({}, {
      get: (target, key) => {
        return this.localeStore.get(`${ID_PREFIX}fields.${key}`) as string;
      }
    });
    return <Row gutter={{xs:0,sm:4}}>
      <Col xs={24} sm={16}>
    <Form className="login-form" >
      <Card>
        <FormItem valid={this.info.titleValid} messageOnInvalid={locale.requireTitle}>
          <Input addonBefore={locale.title}
                 onChange={this.onTitleChange}
                 value={this.info.title}
          />
        </FormItem>
        <FormItem valid={this.info.descriptionValid} messageOnInvalid={locale.requireDescription}>
          <Input.TextArea onChange={this.onDescriptionChange}
                          placeholder={locale.description}
                          value={this.info.description}
          />
        </FormItem>
        <FormItem valid={true} messageOnInvalid={""}>
          <TopicSelector selected={this.info.topics} onChange={this.onTopicChange}/>
        </FormItem>
      </Card>

      <Card >
        <FormItem valid={true} messageOnInvalid={""}>
          <p>{locale.cover}</p>
          <ImageUploadPanel onFileListChange={this.onCoverImageChange}
                          fileList={[this.info.coverImage].filter(x => !!x)}
                          maxFileNum={1}
                          multiple={false}
                          buttonChildren={locale.selectFile}
          />
        </FormItem>
        <FormItem valid={this.info.dateRangeValid} messageOnInvalid={locale.requireDateRange}>
          <p>{locale.dateRange}</p>
          <RangePicker value={toJS(this.info.dateRange)} onChange={this.onDateRangeChanged}/>
        </FormItem>
      </Card>

      <Card >
      <FormItem valid={this.info.minimalWorkerLevelValid}
                messageOnInvalid={locale.requireMinimalWorkerLevel}
                messageOnSuccess={locale.requireMinimalWorkerLevel}
      >
        <Input addonBefore={locale.minimalWorkerLevel}
               onChange={this.onMinimalWorkerLevelChanged}
               value={this.info.minimalWorkerLevel}
        />
      </FormItem>

      <FormItem valid={this.info.levelValid}
                messageOnInvalid={locale.requireMissionLevel}
                messageOnSuccess={locale.requireMissionLevel}
      >
        <Input addonBefore={locale.missionLevel}
               onChange={this.onMissionLevelChanged}
               value={this.info.level}
        />
      </FormItem>

      <CreditInput onChanged={this.onCreditsChanged}/>
      </Card>
      </Form>
      </Col>

      <Col xs={24} sm={8}>
        <Card>
          <MissionTypeSelectPanel info={this.info}
                                  locale={locale}
                                  onAllowCustomTagChanged={this.onAllowCustomTagChanged}
                                  onTagsChange={this.onTagsChange}
                                  onFileListChange={this.onFileListChange}
                                  onTypeChange={this.onTypeChange}
          />
        </Card>
        <Card>
        <Button type={"primary"} onClick={this.submit} loading={this.uploading}>
          {locale.submit}
        </Button>
        </Card>
      </Col>
    </Row>
  }
}
