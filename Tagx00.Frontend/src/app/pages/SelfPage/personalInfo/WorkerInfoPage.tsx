import React from "react";
import {UserStore} from "../../../stores/UserStore";
import {WorkerService} from "../../../api/WorkerService";
import {Inject} from "react.di";
import {LocaleMessage} from "../../../internationalization/components";
import {AsyncComponent} from "../../../router/AsyncComponent";
import {LevelStore} from "../../../stores/LevelStore";

export class WorkerInfoPage extends React.Component<{},{}> {
    @Inject userStore: UserStore;
    @Inject workerService: WorkerService;
    @Inject levelStore: LevelStore;

    workerInfo = async () => {
        const info = await this.workerService.getWorkerInfo(this.userStore.user.username,this.userStore.token);
        return <div>
            <p>用户名：{info.username}</p>
            <p>注册邮箱：{info.email}</p>
            <p>等级：{info.level} //经验进度条</p>
            <p>积分：{info.credits }</p>
        </div>
    }

    render() {
        return (
            <div>
                <h1>
                    <LocaleMessage id={"selfCenter.personalInfo"}/>
                </h1>
                <AsyncComponent render={this.workerInfo}/>
            </div>
        )
    }
}