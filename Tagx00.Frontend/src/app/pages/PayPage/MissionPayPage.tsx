import React from 'react';
import { Inject } from "react.di";
import { Button, Input, Modal } from 'antd';
import { requireLogin } from "../hoc/RequireLogin";
import { UserRole } from "../../models/user/User";
import { FormItem } from "../../components/Form/FormItem";
import { LocaleMessage } from "../../internationalization/components";
import { RequesterService } from "../../api/RequesterService";
import { LocaleStore } from "../../stores/LocaleStore";
import { RichFormItem } from "../../components/Form/RichFormItem";
import { FormItemProps } from "antd/es/form";
import { CreditInput } from "../../components/Pay/CreditInput";

interface Props {
  missionId: string;
  token?: string;
}

interface State {
  missionId: string;
  missionIdCheckOutOfDate: boolean;
  remainingCredits: number;
  fetchingRemainingCredits: boolean;
  value: string;
  key: number;
  inputValid: boolean;
  paying: boolean;
}

const ID_PREFIX = "pay.mission.";

@requireLogin(UserRole.ROLE_REQUESTER)
export class MissionPayPage extends React.Component<Props, State> {

  state = {
    missionId: this.props.missionId || "",
    missionIdCheckOutOfDate: true,
    value: "0",
    inputValid: false,
    key: 0,
    fetchingRemainingCredits: false,
    remainingCredits: 0, // -1 loading, -2 mission not exist, -3 no input
    paying: false,
  };

  @Inject requesterService: RequesterService;
  @Inject localeStore: LocaleStore;

  onMissionIdChanged = (e) => {
    this.setState({
      missionId: e.target.value,
      missionIdCheckOutOfDate: true
    });
  };

  onValueChanged = (value: number, valid: boolean) => {
    this.setState({value: value+"", inputValid: valid });
  };

  loadCurrentCredits = async () => {
    this.setState({ fetchingRemainingCredits: true});
    if (this.state.missionId)
      try {
        const res = await this.requesterService.getRemainingCreditsForAMission(this.state.missionId);
        this.setState({remainingCredits: res.remainingCredits, fetchingRemainingCredits: false,missionIdCheckOutOfDate: false});
      } catch (e) {
        this.setState({remainingCredits: -2, fetchingRemainingCredits: false, missionIdCheckOutOfDate: false});
      }

  };

  componentDidMount() {
    if (this.state.missionId) {
      this.loadCurrentCredits();
    }

  }

  submittable() {
    return  this.payValueValid && this.state.remainingCredits >= 0;
  }


  onSubmit = async () => {
    this.setState({paying: true});
    if (this.state.missionIdCheckOutOfDate) {
      await this.loadCurrentCredits();
    }

    if (this.submittable()) {

      const res = await this.requesterService.payMission(this.state.missionId, parseInt(this.state.value));
      this.setState({remainingCredits: res.remainingCredits, paying: false, key: this.state.key+1});
      Modal.success({
        title: this.localeStore.get(ID_PREFIX + "paymentComplete")
      });

    } else {
      this.setState({paying: false});
    }
  };


  get payValueValid() {
    return this.state.inputValid;
  }

  remainingPrompt = (missionIdStatus: number): FormItemProps => {

    if (!this.state.missionId) {
      return {
        validateStatus: "error",
        help: <LocaleMessage id={ID_PREFIX + "inputMissionId"}/>
      }
    }

    if (this.state.fetchingRemainingCredits) {
      return {
        validateStatus: "validating",
        help: <LocaleMessage id={ID_PREFIX + "loading"}/>
      };
    }

    if (this.state.missionIdCheckOutOfDate) {
      return {
        validateStatus: "success",
        help: <LocaleMessage id={ID_PREFIX + "needCheck"} replacements={{
          click: <a onClick={this.onCheckClicked}><LocaleMessage id={ID_PREFIX + "click"}/></a>
        }}/>
      }
    }

    if (missionIdStatus >= 0) {
      return {
        validateStatus: "success",
        help: <LocaleMessage id={ID_PREFIX + "currentCredits"} replacements={{
          credits: missionIdStatus + ""
        }}/>
      };
    }

    if (missionIdStatus == -2) {
      return {
        validateStatus: "error",
        help: <LocaleMessage id={ID_PREFIX + "missionNotExist"}/>
      }
    }

    return null;

  };


  onCheckClicked = () => {
    this.loadCurrentCredits();
  };


  render() {
    return <div>
      <h1><LocaleMessage id={ID_PREFIX + "title"}/></h1>
      <RichFormItem status={this.state.remainingCredits} mapToFormProps={this.remainingPrompt}>
        <Input addonBefore={<LocaleMessage id={ID_PREFIX + "missionId"}/>}
               value={this.state.missionId}
               onChange={this.onMissionIdChanged}/>
      </RichFormItem>
      <CreditInput key={this.state.key} onChanged={this.onValueChanged}/>
      <Button loading={this.state.paying} type={"primary"} onClick={this.onSubmit}>
        <LocaleMessage id={ID_PREFIX + "submit"}/>
      </Button>
    </div>;
  }
}
