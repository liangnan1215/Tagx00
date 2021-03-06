import React, { ReactNode } from 'react';
import { AsyncComponent } from "../../../router/AsyncComponent";
import { Loading } from "../../Common/Loading/index";
import { Inject } from "react.di";
import { HttpService } from "../../../api/HttpService";
import { HttpMethod } from "../../../api/utils";
import styled from "styled-components";
import { Tabs, Tag as AntdTag} from 'antd';
import { MissionService } from "../../../api/MissionService";
import { LocaleMessage } from "../../../internationalization/components/index";
import { LoadingBarContainer } from "../../../layouts/BaseLayout/LoadingBarContainer";
import { TagMissionContent } from "../../TagMission/index";
import QueueAnim from 'rc-queue-anim';

const { TabPane } = Tabs;

interface Props {
  textToken: string;
  missionId: string;
  onTagClicked?(tag: string): void;
  selectedTags?: string[];
}

const ID_PREFIX = "drawingPad.textReader.";

const Container = styled.div`
  //max-width: 1000px;
  overflow: auto;
  padding: 8px 8px 8px 8px;
  margin-right: auto;
  margin-left: auto;
`;

interface State {
  loading: boolean;
  textToken: string;
}

const TagContainer = styled.div`
  max-height: 600px;
  overflow: auto;
  
  .ant-tag {
  font-size: 14px;
      margin: 4px 2px 4px 2px;
 
  }
`;

interface MyTagProps {
  selected: boolean;
}


interface Cache {
  text: string;
  segmented: string[];
}

const cache: {[s: string]: Cache} = {};

const MyTag = styled.div`
    border-radius: 6px;
    background: ${(props:MyTagProps) => props.selected ? "#108ee9" : "#FAFAFA"};
    //border: 2px solid ${(props:MyTagProps) => props.selected ? "#FFFFFF" : "#D8D8D8"};
    padding: 4px; 
    margin: 4px 2px 4px 2px;
    display: inline-block;
    
    color: ${(props:MyTagProps) => props.selected ? "white" : undefined};
    
    :hover {
      cursor: pointer;
      color: #108ee9;
    }
    
    transition: color 0.3s, background 0.3s;
`;

function Tag(props: {content: string, onClick(): void, selected: boolean}) {

  return <MyTag onClick={props.onClick} selected={props.selected}>
      {props.content}
  </MyTag>;

  // return <AntdTag onClick={props.onClick} color={props.selected ? "#108ee9" : undefined}>
  //   {props.content}
  // </AntdTag>
}

export class SegmentedPane extends React.PureComponent<{ onClick(tag: string): void, words: string[], selectedWords: string[]}> {
  render() {
    return <TagContainer>
      {this.props.words.map((x,i) =>
        <Tag key={i} content={x} selected={this.props.selectedWords.indexOf(x)>=0} onClick={() => this.props.onClick(x)}/>
      )}
    </TagContainer>
  }
}

export class TextReader extends React.Component<Props, State> {

  state = {
    loading: false,
    textToken: ""
  };

  text: string;
  segmented: string[];

  componentDidMount() {
    this.load();
  }

  componentDidUpdate() {
    if (this.props.textToken !== this.state.textToken) {
      this.load();
    }
  }


  async load() {

    this.setState({
      loading: true,
      textToken: this.props.textToken
    });


    const cached = cache[this.props.textToken];
    if (cached) {
      this.text = cached.text;
      this.segmented = cached.segmented;
    } else {
      this.text = await this.missionService.getTextByToken(this.props.textToken);
      this.segmented = await this.missionService.segmentWord(this.props.textToken, this.props.missionId);
      cache[this.props.textToken] = { text: this.text, segmented: this.segmented};
    }

    this.setState({
      loading: false
    });

  }


  static defaultProps = {
    selectedTags: []
  };

  @Inject missionService: MissionService;

  onTagClick = (tag: string) => {
    this.props.onTagClicked && this.props.onTagClicked(tag);
  };

  render() {
    if (this.state.loading) {
      return <Loading/>;
    }
    return <Container>
      <Tabs>
        <TabPane key={"raw"} tab={<LocaleMessage id={ID_PREFIX+"raw"}/>}>
          {this.text}
        </TabPane>
        <TabPane key={"segmented"} tab={<LocaleMessage id={ID_PREFIX+"segmented"}/>}>
          <SegmentedPane onClick={this.onTagClick}
                         words={this.segmented}
                         selectedWords={this.props.selectedTags}/>
        </TabPane>
      </Tabs>
    </Container>
  }
}
