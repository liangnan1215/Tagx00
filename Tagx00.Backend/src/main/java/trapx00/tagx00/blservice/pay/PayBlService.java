package trapx00.tagx00.blservice.pay;

import trapx00.tagx00.exception.viewexception.SystemException;
import trapx00.tagx00.exception.viewexception.UserDoesNotExistException;
import trapx00.tagx00.response.pay.PayQueryResponse;
import trapx00.tagx00.response.pay.PayResponse;
import trapx00.tagx00.vo.mission.pay.PayVo;

public interface PayBlService {

    /**
     * pay for account
     *
     * @param payVo
     * @param username
     * @return PayResponse
     */
    PayResponse pay(PayVo payVo, String username) throws SystemException;

    /**
     * query the credits the user now has
     *
     * @param username the username
     * @return the credits
     */
    PayQueryResponse queryPay(String username) throws UserDoesNotExistException;

}
