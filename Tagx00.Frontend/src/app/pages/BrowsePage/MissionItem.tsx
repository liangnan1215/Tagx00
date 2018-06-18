import React from 'react';
import { Button, List, Tag } from 'antd';
import { Inject } from "react.di";
import { WorkerService } from "../../api/WorkerService";
import { UserStore } from "../../stores/UserStore";
import { Link } from 'react-router-dom';
import { LocaleDate, LocaleMessage } from "../../internationalization/components";
import { RouterStore } from "../../stores/RouterStore";
import styled from "styled-components";
import { MissionPublicItem } from "../../models/mission/MissionPublicItem";
import { DEFAULT_COVER_URL } from "../../components/Mission/util";

interface Props {
  item: MissionPublicItem;
}

const ID_PREFIX = "browserMissionList.";

function processTags(topics: string[], missionType: string) {
  return [
    <Tag key={"_type"} color="#108ee9"><LocaleMessage id={`${ID_PREFIX}missionType.${missionType}`}/></Tag>,
    ...topics.map(x => <Tag key={x} color="geekblue">{x}</Tag>)
  ];
}

const Row = styled.div`
  display: flex;
  align-items: center;
`;

const Img = styled.img`
  margin-right: 16px;
  width: 200px;
  
  @media (max-width: 450px) {
  width:100px;
  }
`;

const MetaContainer = styled.div`
  min-width: 300px;
`;



export class MissionItem extends React.PureComponent<Props, {}> {

  @Inject workerService: WorkerService;
  @Inject userStore: UserStore;
  @Inject routerStore: RouterStore; // link within actions causes error, don't know why

  jumpToDetail = () => {
    const toLink = `/mission?missionId=${this.props.item.missionId}`;
    this.routerStore.jumpTo(toLink);
  };

  render() {
    const {item} = this.props;
    return <Row style={{marginBottom: "8px"}}>
          <Img alt="logo" src={item.coverUrl || DEFAULT_COVER_URL}/>
      <MetaContainer>
        <List.Item
          actions={[
            <Button key={"seeDetail"} type="primary" icon="info" onClick={this.jumpToDetail}>
              <LocaleMessage id={ID_PREFIX + "seeDetail"}/>
            </Button>
          ]}
        >

          <List.Item.Meta
            title={<a onClick={this.jumpToDetail}>{item.title}</a>}
            description={
              <>
                {processTags(item.topics, item.missionType)}
                <span>
              <LocaleDate formatId={ID_PREFIX + "dateFormat"} input={item.start}/>
                  &nbsp;-&nbsp;
                  <LocaleDate formatId={ID_PREFIX + "dateFormat"} input={item.end}/>
            </span>
              </>
            }/>
          {item.description}
        </List.Item>
      </MetaContainer>
    </Row>
  }
}
