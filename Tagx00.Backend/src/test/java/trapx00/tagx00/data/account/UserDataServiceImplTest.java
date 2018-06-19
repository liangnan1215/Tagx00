package trapx00.tagx00.data.account;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.test.context.junit4.SpringRunner;
import trapx00.tagx00.data.dao.mission.MissionDao;
import trapx00.tagx00.data.dao.mission.topic.TopicDao;
import trapx00.tagx00.data.dao.user.UserDao;
import trapx00.tagx00.dataservice.account.UserDataService;
import trapx00.tagx00.entity.account.Role;
import trapx00.tagx00.entity.account.User;
import trapx00.tagx00.entity.mission.*;
import trapx00.tagx00.entity.mission.topic.Topic;
import trapx00.tagx00.publicdatas.mission.MissionState;
import trapx00.tagx00.publicdatas.mission.MissionType;
import trapx00.tagx00.util.MissionUtil;
import trapx00.tagx00.vo.mission.audio.AudioMissionType;
import trapx00.tagx00.vo.mission.image.ImageMissionType;
import trapx00.tagx00.vo.mission.text.TextMissionType;
import trapx00.tagx00.vo.mission.threedimension.ThreeDimensionMissionType;
import trapx00.tagx00.vo.mission.video.VideoMissionType;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

@RunWith(SpringRunner.class)
@SpringBootTest
public class UserDataServiceImplTest {
    @Autowired
    private UserDataService userDataService;
    @Autowired
    private UserDao userDao;
    @Autowired
    private MissionDao missionDao;
    @Autowired
    private TopicDao topicDao;
    private Random random = new Random();
    private SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private java.util.Date startDate = format.parse("2018-4-1 12:00:00");
    private java.util.Date notEndedDate = format.parse("2018-7-10 12:00:00");
    private java.util.Date endDate = new java.util.Date();

    public UserDataServiceImplTest() throws ParseException {
    }

    @Test
    public void isUserExistent() {
    }

    @Test
    public void addTopics() {
        Topic topic = new Topic("动物");
        topicDao.save(topic);
        topic = new Topic("植物");
        topicDao.save(topic);
        topic = new Topic("日常用品");
        topicDao.save(topic);
        topic = new Topic("家居用品");
        topicDao.save(topic);
        topic = new Topic("生活用品");
        topicDao.save(topic);
    }

    @Test
    public void saveUser() {
        BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();
        String username = "999";
        String password = encoder.encode("999");
        String email = "445073309@qq.com";
        Role role = Role.ADMIN;
        double exp = 0;
        int credits = 0;

        java.sql.Date sqlDate = new java.sql.Date(endDate.getTime());
        User user = new User(username, password, email, role, exp, credits, sqlDate);
        userDao.save(user);
        String randomInfo;
        for (int i = 0; i < 100; i++) {
            randomInfo = random.nextInt(10000) + "";
            username = randomInfo;
            password = encoder.encode(randomInfo);
            role = Role.REQUESTER;
            if (userDao.findUserByUsername(username) == null) {
                exp = 0;
                credits = random.nextInt(200000);
                long date = random(startDate.getTime(), endDate.getTime());
                sqlDate = new java.sql.Date(date);
                user = new User(username, password, email, role, exp, credits, sqlDate);
                userDao.save(user);

                List<ImageMissionType> imageMissionTypes = new ArrayList<>();
                imageMissionTypes.add(ImageMissionType.WHOLE);
                imageMissionTypes.add(ImageMissionType.DISTRICT);
                imageMissionTypes.add(ImageMissionType.PART);
                ImageMission imageMission = new ImageMission(getNextId(missionDao.findAll(), MissionType.IMAGE), "test", "test", new ArrayList<>(), MissionState.ACTIVE, startDate, endDate, "", username,
                        1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), imageMissionTypes, new ArrayList<>());
                missionDao.save(imageMission);

                for (int j = 0; j < 6; j++) {
                    int selectSeed = random.nextInt(6);
                    switch (selectSeed) {
                        case 1:
                            int selectCreate = random.nextInt(4);
                            switch (selectCreate) {
                                case 1:
                                    createImageMission(username, MissionState.PENDING);
                                    break;
                                case 2:
                                    createImageMission(username, MissionState.ACTIVE);
                                    break;
                                default:
                                    createImageMission(username, MissionState.ENDED);
                            }
                            break;
                        case 2:
                            selectCreate = random.nextInt(4);
                            switch (selectCreate) {
                                case 1:
                                    createAudioMission(username, MissionState.PENDING);
                                    break;
                                case 2:
                                    createAudioMission(username, MissionState.ACTIVE);
                                    break;
                                default:
                                    createAudioMission(username, MissionState.ENDED);
                            }
                            break;
                        case 3:
                            selectCreate = random.nextInt(4);
                            switch (selectCreate) {
                                case 1:
                                    createVideoMission(username, MissionState.PENDING);
                                    break;
                                case 2:
                                    createVideoMission(username, MissionState.ACTIVE);
                                    break;
                                default:
                                    createVideoMission(username, MissionState.ENDED);
                            }
                            break;
                        case 4:
                            selectCreate = random.nextInt(4);
                            switch (selectCreate) {
                                case 1:
                                    createTextMission(username, MissionState.PENDING);
                                    break;
                                case 2:
                                    createTextMission(username, MissionState.ACTIVE);
                                    break;
                                default:
                                    createTextMission(username, MissionState.ENDED);
                            }
                            break;
                        case 5:
                            selectCreate = random.nextInt(4);
                            switch (selectCreate) {
                                case 1:
                                    createThreeDimensionMission(username, MissionState.PENDING);
                                    break;
                                case 2:
                                    createThreeDimensionMission(username, MissionState.ACTIVE);
                                    break;
                                default:
                                    createThreeDimensionMission(username, MissionState.ENDED);
                            }
                            break;
                    }
                }
            }
        }

        List<Mission> missions = missionDao.findAll();
        for (int i = 0; i < 500; i++) {
            randomInfo = random.nextInt(10000) + "";
            username = randomInfo;
            password = encoder.encode(randomInfo);
            role = Role.WORKER;
            if (userDao.findUserByUsername(username) == null) {
                exp = random.nextInt(1000);
                credits = random.nextInt(100000);
                long date = random(startDate.getTime(), endDate.getTime());
                sqlDate = new java.sql.Date(date);
                user = new User(username, password, email, role, exp, credits, sqlDate);
                userDao.save(user);
            }
        }
    }

    private static long random(long begin, long end) {
        long rtn = begin + (long) (Math.random() * (end - begin));
        // 如果返回的是开始时间和结束时间，则递归调用本函数查找随机值
        if (rtn == begin || rtn == end) {
            return random(begin, end);
        }
        return rtn;
    }

    private void createVideoMission(String username, MissionState missionState) {
        List<VideoMissionType> videoMissionTypes = new ArrayList<>();
        int typeSelect = random.nextInt(3);
        switch (typeSelect) {
            case 1:
                videoMissionTypes.add(VideoMissionType.WHOLE);
                break;
            case 2:
                videoMissionTypes.add(VideoMissionType.PART);
                break;
            default:
                videoMissionTypes.add(VideoMissionType.WHOLE);
                videoMissionTypes.add(VideoMissionType.PART);
        }
        VideoMission videoMission = null;
        if (MissionState.PENDING == missionState) {
            videoMission = new VideoMission(getNextId(missionDao.findAll(), MissionType.VIDEO), "test", "test", new ArrayList<>(), missionState, notEndedDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), videoMissionTypes, new ArrayList<>());
        }
        if (MissionState.ENDED == missionState) {
            videoMission = new VideoMission(getNextId(missionDao.findAll(), MissionType.VIDEO), "test", "test", new ArrayList<>(), missionState, startDate, startDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), videoMissionTypes, new ArrayList<>());
        }
        if (MissionState.ACTIVE == missionState) {
            videoMission = new VideoMission(getNextId(missionDao.findAll(), MissionType.VIDEO), "test", "test", new ArrayList<>(), missionState, startDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), videoMissionTypes, new ArrayList<>());
        }
        missionDao.save(videoMission);
    }

    private void createAudioMission(String username, MissionState missionState) {
        List<AudioMissionType> audioMissionTypes = new ArrayList<>();
        int typeSelect = random.nextInt(3);
        switch (typeSelect) {
            case 1:
                audioMissionTypes.add(AudioMissionType.WHOLE);
                break;
            case 2:
                audioMissionTypes.add(AudioMissionType.PART);
                break;
            default:
                audioMissionTypes.add(AudioMissionType.WHOLE);
                audioMissionTypes.add(AudioMissionType.PART);
        }
        AudioMission audioMission = null;
        if (MissionState.PENDING == missionState) {
            audioMission = new AudioMission(getNextId(missionDao.findAll(), MissionType.AUDIO), "test", "test", new ArrayList<>(), missionState, notEndedDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), audioMissionTypes, new ArrayList<>());
        }
        if (MissionState.ENDED == missionState) {
            audioMission = new AudioMission(getNextId(missionDao.findAll(), MissionType.AUDIO), "test", "test", new ArrayList<>(), missionState, startDate, startDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), audioMissionTypes, new ArrayList<>());
        }
        if (MissionState.ACTIVE == missionState) {
            audioMission = new AudioMission(getNextId(missionDao.findAll(), MissionType.AUDIO), "test", "test", new ArrayList<>(), missionState, startDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), audioMissionTypes, new ArrayList<>());
        }
        missionDao.save(audioMission);
    }

    private void createThreeDimensionMission(String username, MissionState missionState) {
        List<ThreeDimensionMissionType> threeDimensionMissionTypes = new ArrayList<>();
        threeDimensionMissionTypes.add(ThreeDimensionMissionType.WHOLE);
        ThreeDimensionMission threeDimensionMission = null;
        if (MissionState.PENDING == missionState) {
            threeDimensionMission = new ThreeDimensionMission(getNextId(missionDao.findAll(), MissionType.THREE_DIMENSION), "test", "test", new ArrayList<>(), missionState, notEndedDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), ThreeDimensionMissionType.WHOLE, new ArrayList<>());
        }
        if (MissionState.ENDED == missionState) {
            threeDimensionMission = new ThreeDimensionMission(getNextId(missionDao.findAll(), MissionType.THREE_DIMENSION), "test", "test", new ArrayList<>(), missionState, startDate, startDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), ThreeDimensionMissionType.WHOLE, new ArrayList<>());
        }
        if (MissionState.ACTIVE == missionState) {
            threeDimensionMission = new ThreeDimensionMission(getNextId(missionDao.findAll(), MissionType.THREE_DIMENSION), "test", "test", new ArrayList<>(), missionState, startDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), ThreeDimensionMissionType.WHOLE, new ArrayList<>());
        }
        missionDao.save(threeDimensionMission);
    }

    private void createTextMission(String username, MissionState missionState) {
        List<TextMissionType> textMissionTypes = new ArrayList<>();
        int typeSelect = random.nextInt(3);
        switch (typeSelect) {
            case 1:
                textMissionTypes.add(TextMissionType.CLASSIFICATION);
                break;
            case 2:
                textMissionTypes.add(TextMissionType.KEYWORDS);
                break;
            default:
                textMissionTypes.add(TextMissionType.CLASSIFICATION);
                textMissionTypes.add(TextMissionType.KEYWORDS);
        }
        TextMission textMission = null;
        if (MissionState.PENDING == missionState) {
            textMission = new TextMission(getNextId(missionDao.findAll(), MissionType.TEXT), "test", "test", new ArrayList<>(), missionState, notEndedDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, new HashSet<>(), new ArrayList<>(), new ArrayList<>());
        }
        if (MissionState.ENDED == missionState) {
            textMission = new TextMission(getNextId(missionDao.findAll(), MissionType.TEXT), "test", "test", new ArrayList<>(), missionState, startDate, startDate, "", username,
                    1, random.nextInt(10000), 1, new HashSet<>(), new ArrayList<>(), new ArrayList<>());
        }
        if (MissionState.ACTIVE == missionState) {
            textMission = new TextMission(getNextId(missionDao.findAll(), MissionType.TEXT), "test", "test", new ArrayList<>(), missionState, startDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, new HashSet<>(), new ArrayList<>(), new ArrayList<>());
        }
        missionDao.save(textMission);
    }

    private void createImageMission(String username, MissionState missionState) {
        List<ImageMissionType> imageMissionTypes = new ArrayList<>();
        int typeSelect = random.nextInt(4);
        switch (typeSelect) {
            case 1:
                imageMissionTypes.add(ImageMissionType.WHOLE);
                break;
            case 2:
                imageMissionTypes.add(ImageMissionType.DISTRICT);
                imageMissionTypes.add(ImageMissionType.PART);
                break;
            case 3:
                imageMissionTypes.add(ImageMissionType.WHOLE);
                imageMissionTypes.add(ImageMissionType.PART);
                break;
            default:
                imageMissionTypes.add(ImageMissionType.WHOLE);
                imageMissionTypes.add(ImageMissionType.DISTRICT);
        }
        ImageMission imageMission = null;
        if (MissionState.PENDING == missionState) {
            imageMission = new ImageMission(getNextId(missionDao.findAll(), MissionType.IMAGE), "test", "test", new ArrayList<>(), missionState, notEndedDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), imageMissionTypes, new ArrayList<>());
        }
        if (MissionState.ENDED == missionState) {
            imageMission = new ImageMission(getNextId(missionDao.findAll(), MissionType.IMAGE), "test", "test", new ArrayList<>(), missionState, startDate, startDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), imageMissionTypes, new ArrayList<>());
        }
        if (MissionState.ACTIVE == missionState) {
            imageMission = new ImageMission(getNextId(missionDao.findAll(), MissionType.IMAGE), "test", "test", new ArrayList<>(), missionState, startDate, notEndedDate, "", username,
                    1, random.nextInt(10000), 1, false, new ArrayList<>(), new ArrayList<>(), imageMissionTypes, new ArrayList<>());
        }
        missionDao.save(imageMission);
    }

    private <T extends Mission> String getNextId(List<T> missions, MissionType missionType) {
        int result = 0;
        Optional<T> latestMission = missions.stream().max((x1, x2) -> (MissionUtil.getId(x1.getMissionId()) - MissionUtil.getId(x2.getMissionId())));
        if (latestMission.isPresent()) {
            result = MissionUtil.getId(latestMission.get().getMissionId()) + 1;
        }
        return MissionUtil.addTypeToId(result, missionType);
    }

    @Test
    public void confirmPassword() {
    }

    @Test
    public void deleteUser() {
    }

    @Test
    public void sendEmail() {
//        userDataService.sendEmail("445073309@qq.com","445073309@qq.com");
    }
}